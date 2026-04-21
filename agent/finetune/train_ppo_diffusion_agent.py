"""
DPPO fine-tuning.

"""

from __future__ import annotations

import os
import pickle
import time
import einops
import numpy as np
import torch
import logging
import wandb
import math

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_agent import TrainPPOAgent
from util.scheduler import CosineAnnealingWarmupRestarts


def _format_hhmmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def _format_iter_summary(
    *,
    it: int,
    total_it: int,
    start_it: int,
    tot_time: float,
    tot_timesteps: int,
    collect_time: float,
    learn_time: float,
    n_steps: int,
    n_envs: int,
    act_steps: int,
    eval_mode: bool,
    loss: float,
    pg_loss: float,
    v_loss: float,
    bc_loss: float,
    eta: float,
    approx_kl: float,
    clipfrac: float,
    explained_var: float,
    ratio: float,
    avg_episode_reward: float,
    avg_best_reward: float,
    success_rate: float,
    num_episode_finished: int,
    mean_episode_length: float,
    actor_lr: float,
    critic_lr: float,
    diffusion_min_sampling_std: float,
    width: int = 80,
    pad: int = 40,
) -> str:
    """Render an RSL-RL-style boxed per-iteration summary."""
    iter_time = collect_time + learn_time
    iter_env_steps = n_steps * n_envs * act_steps
    fps = iter_env_steps / iter_time if iter_time > 0 else 0
    done_it = it + 1 - start_it
    remaining_it = max(0, total_it - start_it - done_it)
    eta_wall = tot_time / done_it * remaining_it if done_it > 0 else 0.0

    tag = "Eval iteration" if eval_mode else "Learning iteration"
    lines = []
    lines.append("#" * width)
    lines.append(f"\033[1m{f' {tag} {it}/{total_it} '.center(width)}\033[0m")
    lines.append("")
    lines.append(f"{'Total steps:':>{pad}} {tot_timesteps}")
    lines.append(f"{'Steps per second:':>{pad}} {fps:.0f}")
    lines.append(f"{'Collection time:':>{pad}} {collect_time:.3f}s")
    lines.append(f"{'Learning time:':>{pad}} {learn_time:.3f}s")
    lines.append(f"{'Loss:':>{pad}} {loss:.4f}")
    lines.append(f"{'PG loss:':>{pad}} {pg_loss:.4f}")
    lines.append(f"{'Value loss:':>{pad}} {v_loss:.4f}")
    lines.append(f"{'BC loss:':>{pad}} {bc_loss:.4f}")
    lines.append(f"{'Eta (entropy-ish):':>{pad}} {eta:.4f}")
    lines.append(f"{'Approx KL:':>{pad}} {approx_kl:.4f}")
    lines.append(f"{'Clip fraction:':>{pad}} {clipfrac:.4f}")
    lines.append(f"{'Explained variance:':>{pad}} {explained_var:.4f}")
    lines.append(f"{'Policy ratio:':>{pad}} {ratio:.4f}")
    lines.append(f"{'Mean episode reward:':>{pad}} {avg_episode_reward:.3f}")
    lines.append(f"{'Mean best-step reward:':>{pad}} {avg_best_reward:.3f}")
    lines.append(f"{'Success rate:':>{pad}} {success_rate:.3f}")
    lines.append(f"{'Mean episode length:':>{pad}} {mean_episode_length:.1f}")
    lines.append(f"{'Episodes completed:':>{pad}} {num_episode_finished}")
    lines.append(f"{'Actor LR:':>{pad}} {actor_lr:.6g}")
    lines.append(f"{'Critic LR:':>{pad}} {critic_lr:.6g}")
    lines.append(
        f"{'Diffusion min sampling std:':>{pad}} {diffusion_min_sampling_std:.4f}"
    )
    lines.append("-" * width)
    lines.append(f"{'Iteration time:':>{pad}} {iter_time:.2f}s")
    lines.append(f"{'Time elapsed:':>{pad}} {_format_hhmmss(tot_time)}")
    lines.append(f"{'ETA:':>{pad}} {_format_hhmmss(eta_wall)}")
    return "\n".join(lines)


def _coerce_success_flag(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(_coerce_success_flag(elem) for elem in value)
    if isinstance(value, np.ndarray):
        return bool(np.max(value))
    return bool(value)


class TrainPPODiffusionAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Default mode only -- reduce-overhead (CUDA graphs) errors on
        # overwritten output tensors shared across denoising loop + PPO update.
        if bool(getattr(cfg.train, "torch_compile", True)):
            try:
                self.model.actor = torch.compile(self.model.actor)
                self.model.actor_ft = torch.compile(self.model.actor_ft)
                self.model.critic = torch.compile(self.model.critic)
                log.info("torch.compile enabled (mode=default) for actor/actor_ft/critic")
            except Exception as exc:
                log.warning(f"torch.compile failed, falling back to eager: {exc}")

        # Reward horizon --- always set to act_steps for now
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)

        # Warp venv accepts torch-on-device directly; CPU AsyncVectorEnv doesn't.
        venv_dev = getattr(self.venv, "device", None)
        self._venv_accepts_torch = (
            isinstance(venv_dev, torch.device)
            and venv_dev.type == torch.device(self.device).type
            and venv_dev.index == torch.device(self.device).index
        )

        # Eta - between DDIM (=0 for eval) and DDPM (=1 for training)
        self.learn_eta = self.model.learn_eta
        if self.learn_eta:
            self.eta_update_interval = cfg.train.eta_update_interval
            self.eta_optimizer = torch.optim.AdamW(
                self.model.eta.parameters(),
                lr=cfg.train.eta_lr,
                weight_decay=cfg.train.eta_weight_decay,
            )
            self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.eta_optimizer,
                first_cycle_steps=cfg.train.eta_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.eta_lr,
                min_lr=cfg.train.eta_lr_scheduler.min_lr,
                warmup_steps=cfg.train.eta_lr_scheduler.warmup_steps,
                gamma=1.0,
            )

    def _collect_iter_videos(self) -> dict[str, wandb.Video]:
        """Return {video[...]: wandb.Video(...)} for MP4s written this iter.

        Keyed under a single top-level ``video`` panel (``video`` when only
        one render env, else ``video_k``) so eval and rollout recordings
        share the same wandb panel instead of being split across
        ``Eval/video_0`` / ``Rollout/video_0``.

        No-ops when video recording is off, wandb is off, or this iter is not
        a render iter.
        """
        if not self.use_wandb or not self.render_video:
            return {}
        if self.itr % self.render_freq != 0:
            return {}
        # wandb.Video copies on construction; imageio writes the moov atom only
        # on close(). Finalize writers first or wandb uploads truncated mp4s.
        close_fn = getattr(self.venv, "close_videos", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                log.warning(f"venv.close_videos() failed: {exc}")
        out: dict[str, wandb.Video] = {}
        for env_ind in range(self.n_render):
            path = os.path.join(
                self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
            )
            if not os.path.exists(path):
                continue
            key = "video" if self.n_render == 1 else f"video_{env_ind}"
            try:
                out[key] = wandb.Video(path, format="mp4")
            except Exception as exc:
                log.warning(f"Failed to wrap video {path} for wandb: {exc}")
        return out

    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        # Boxed summary controls (defaults: boxed on, per-step/per-batch off).
        log_cfg = getattr(self.cfg.train, "log", None)
        self._verbose_rollout = bool(
            log_cfg.get("verbose_rollout", False) if log_cfg is not None else False
        )
        self._verbose_batch = bool(
            log_cfg.get("verbose_batch", False) if log_cfg is not None else False
        )
        self._boxed_summary = bool(
            log_cfg.get("boxed_summary", True) if log_cfg is not None else True
        )
        self._start_itr = self.itr
        self._tot_timesteps = 0
        self._tot_wall_time = 0.0
        while self.itr < self.n_train_itr:
            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            # GPU-resident buffers -- ~700 MB at n_envs=1024; avoids D2H/H2D
            # round trips per rollout step and at the PPO update head.
            obs_trajs_gpu = torch.zeros(
                (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim),
                dtype=torch.float32,
                device=self.device,
            )
            chains_trajs_gpu = torch.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            # Numpy-native downstream (episode slicing, reward scaling).
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            success_trajs = (
                np.zeros((self.n_steps, self.n_envs), dtype=bool)
                if self.success_info_key is not None
                else None
            )
            if self.save_full_observations:  # state-only
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
                )

            # Time rollout (collection) separately from the update phase so
            # the boxed per-iter summary can report both.
            rollout_start = time.time()

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if self._verbose_rollout and step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    cond_state_t = torch.as_tensor(
                        prev_obs_venv["state"], dtype=torch.float32, device=self.device
                    )
                    cond = {"state": cond_state_t}
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    # Warp venv takes torch-on-device; CPU AsyncVectorEnv needs numpy.
                    action_chunk_t = samples.trajectories[:, : self.act_steps]
                    if self._venv_accepts_torch:
                        action_venv = action_chunk_t
                    else:
                        action_venv = action_chunk_t.detach().cpu().numpy()

                # Apply multi-step action
                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    info_venv,
                ) = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                if success_trajs is not None:
                    success_trajs[step] = np.asarray(
                        [
                            _coerce_success_flag(info.get(self.success_info_key))
                            for info in info_venv
                        ],
                        dtype=bool,
                    )
                if self.save_full_observations:  # state-only
                    obs_full_venv = np.array(
                        [info["full_obs"]["state"] for info in info_venv]
                    )  # n_envs x act_steps x obs_dim
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                    )
                # GPU-resident writes -- no host copy.
                obs_trajs_gpu[step] = cond_state_t
                chains_trajs_gpu[step] = samples.chains
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            collect_time = time.time() - rollout_start

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                if (
                    self.furniture_sparse_reward
                ):  # only for furniture tasks, where reward only occurs in one env step
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [
                            np.max(reward_traj) / self.act_steps
                            for reward_traj in reward_trajs_split
                        ]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                if success_trajs is not None:
                    episode_success = np.array(
                        [
                            np.max(success_trajs[start : end + 1, env_ind])
                            for env_ind, start, end in episodes_start_end
                        ],
                        dtype=bool,
                    )
                    success_rate = np.mean(episode_success)
                else:
                    success_rate = np.mean(
                        episode_best_reward >= self.best_reward_threshold_for_success
                    )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
            learn_start = time.time()
            # Initialise loss/update metrics so the summary still renders
            # cleanly in eval-only iters (no update happens).
            loss = pg_loss = v_loss = bc_loss = 0.0
            eta = approx_kl = ratio = 0.0
            explained_var = float("nan")
            # GPU-resident per-batch accumulators -- one sync/epoch, not per-batch.
            _agg_keys = (
                "loss", "pg", "v", "bc", "eta_", "kl", "kl_max", "ratio",
                "actor_gn", "critic_gn", "clipfrac",
            )
            _agg_t = {k: torch.zeros((), device=self.device) for k in _agg_keys}
            _agg_t["kl_max"] = torch.zeros((), device=self.device)
            _agg_count = 0
            if not eval_mode:
                with torch.no_grad():
                    # Calculate value and logprobs - split into batches to prevent out of memory.
                    # All buffers stay on GPU; no vstack/np round-trips.
                    obs_k_t = einops.rearrange(
                        obs_trajs_gpu, "s e ... -> (s e) ..."
                    )
                    chains_k_t = einops.rearrange(
                        chains_trajs_gpu, "s e t h d -> (s e) t h d"
                    )
                    obs_ts_k = torch.split(obs_k_t, self.logprob_batch_size, dim=0)
                    chains_ts = torch.split(chains_k_t, self.logprob_batch_size, dim=0)
                    values_chunks: list[torch.Tensor] = []
                    for obs_t in obs_ts_k:
                        values_chunks.append(
                            self.model.critic({"state": obs_t}).view(-1)
                        )
                    values_flat = torch.cat(values_chunks, dim=0)
                    values_trajs_t = values_flat.view(self.n_steps, self.n_envs)

                    # Reshape (B * ft_denoising_steps, Ta, Da) -> (B, ft_denoising_steps, Ta, Da)
                    # so the update loop can index by (batch_inds, denoising_inds).
                    logprobs_chunks: list[torch.Tensor] = []
                    for obs_t, chains in zip(obs_ts_k, chains_ts):
                        lp = self.model.get_logprobs({"state": obs_t}, chains)
                        logprobs_chunks.append(
                            lp.reshape(
                                obs_t.shape[0],
                                self.model.ft_denoising_steps,
                                self.horizon_steps,
                                self.action_dim,
                            )
                        )
                    logprobs_k = torch.cat(logprobs_chunks, dim=0)

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # GAE on GPU; nextvalues bootstrap from critic(obs_venv).
                    reward_trajs_t = torch.as_tensor(
                        reward_trajs, dtype=torch.float32, device=self.device
                    )
                    nonterminal_t = 1.0 - torch.as_tensor(
                        terminated_trajs, dtype=torch.float32, device=self.device
                    )
                    obs_venv_t = torch.as_tensor(
                        obs_venv["state"], dtype=torch.float32, device=self.device
                    )
                    boot_values = (
                        self.model.critic({"state": obs_venv_t}).view(self.n_envs)
                    )
                    advantages_trajs_t = torch.zeros_like(reward_trajs_t)
                    lastgaelam = torch.zeros(self.n_envs, device=self.device)
                    gamma = self.gamma
                    lam = self.gae_lambda
                    rscale = self.reward_scale_const
                    for t in range(self.n_steps - 1, -1, -1):
                        if t == self.n_steps - 1:
                            nextvalues = boot_values
                        else:
                            nextvalues = values_trajs_t[t + 1]
                        nterm = nonterminal_t[t]
                        delta = (
                            reward_trajs_t[t] * rscale
                            + gamma * nextvalues * nterm
                            - values_trajs_t[t]
                        )
                        lastgaelam = delta + gamma * lam * nterm * lastgaelam
                        advantages_trajs_t[t] = lastgaelam
                    returns_trajs_t = advantages_trajs_t + values_trajs_t

                # k for environment step -- everything already on GPU.
                obs_k = {"state": obs_k_t}
                chains_k = chains_k_t
                returns_k = returns_trajs_t.reshape(-1)
                values_k = values_trajs_t.reshape(-1)
                advantages_k = advantages_trajs_t.reshape(-1)

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps
                for update_epoch in range(self.update_epochs):
                    # for each epoch, go through all data in batches
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        batch_inds_b, denoising_inds_b = torch.unravel_index(
                            inds_b,
                            (self.n_steps * self.n_envs, self.model.ft_denoising_steps),
                        )
                        obs_b = {"state": obs_k["state"][batch_inds_b]}
                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_k[batch_inds_b]
                        values_b = values_k[batch_inds_b]
                        advantages_b = advantages_k[batch_inds_b]
                        logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                        # get loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_prev_b,
                            chains_next_b,
                            denoising_inds_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )

                        # update policy and critic
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        if self.learn_eta:
                            self.eta_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            # max_norm=inf -> returns pre-clip norm without clipping.
                            clip_actor = self.max_grad_norm if self.max_grad_norm is not None else float("inf")
                            actor_gn = torch.nn.utils.clip_grad_norm_(
                                self.model.actor_ft.parameters(), clip_actor
                            )
                            self.actor_optimizer.step()
                            if self.learn_eta and batch % self.eta_update_interval == 0:
                                self.eta_optimizer.step()
                        else:
                            actor_gn = torch.zeros((), device=self.device)
                        critic_gn = torch.nn.utils.clip_grad_norm_(
                            self.model.critic.parameters(), float("inf")
                        )
                        self.critic_optimizer.step()

                        # Hot loop -- avoid .item()/float(); only approx_kl syncs
                        # (for target_kl early-break).
                        with torch.no_grad():
                            _agg_t["loss"] = _agg_t["loss"] + loss.detach()
                            _agg_t["pg"] = _agg_t["pg"] + pg_loss.detach()
                            _agg_t["v"] = _agg_t["v"] + v_loss.detach()
                            if isinstance(bc_loss, torch.Tensor):
                                _agg_t["bc"] = _agg_t["bc"] + bc_loss.detach()
                            _agg_t["eta_"] = _agg_t["eta_"] + eta.detach()
                            _agg_t["kl"] = _agg_t["kl"] + approx_kl.detach()
                            _agg_t["kl_max"] = torch.maximum(
                                _agg_t["kl_max"], approx_kl.detach()
                            )
                            _agg_t["ratio"] = _agg_t["ratio"] + ratio.detach()
                            _agg_t["actor_gn"] = _agg_t["actor_gn"] + actor_gn.detach()
                            _agg_t["critic_gn"] = _agg_t["critic_gn"] + critic_gn.detach()
                            _agg_t["clipfrac"] = _agg_t["clipfrac"] + clipfrac.detach()
                        _agg_count += 1

                        if self._verbose_batch or self.target_kl is not None:
                            kl_val = float(approx_kl)
                            if self._verbose_batch:
                                log.info(
                                    f"approx_kl: {kl_val}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                                )
                            if self.target_kl is not None and kl_val > self.target_kl:
                                flag_break = True
                                break
                    if flag_break:
                        break

                # Explained variation -- computed on GPU, one scalar sync.
                with torch.no_grad():
                    var_y = returns_k.var(unbiased=False)
                    explained_var_t = torch.where(
                        var_y > 0,
                        1.0 - (returns_k - values_k).var(unbiased=False) / var_y,
                        torch.tensor(float("nan"), device=self.device),
                    )
                explained_var = float(explained_var_t)

            # Plot state trajectories (only in D3IL)
            if (
                self.itr % self.render_freq == 0
                and self.n_render > 0
                and self.traj_plotter is not None
            ):
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )

            # Update lr, min_sampling_std
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = {
                    "state": obs_trajs_gpu.detach().cpu().numpy()
                }
                run_results[-1]["chains_trajs"] = chains_trajs_gpu.detach().cpu().numpy()
                run_results[-1]["reward_trajs"] = reward_trajs
            learn_time = 0.0 if eval_mode else (time.time() - learn_start)
            iter_time = collect_time + learn_time
            self._tot_timesteps = cnt_train_step
            self._tot_wall_time += iter_time

            # Mean episode length in env steps (act_steps per outer step).
            if len(episodes_start_end) > 0:
                episode_lengths = np.array(
                    [(end - start + 1) * self.act_steps for _, start, end in episodes_start_end]
                )
                mean_episode_length = float(np.mean(episode_lengths))
            else:
                mean_episode_length = 0.0

            # Rollout diagnostics.
            rollout_mean_reward_per_step = float(reward_trajs.mean())
            rollout_terminated_rate = float(terminated_trajs.mean())
            done_trajs = firsts_trajs[1:].astype(bool)
            rollout_truncated_rate = float(
                (done_trajs & ~terminated_trajs.astype(bool)).mean()
            )
            # Emitted actions = final horizon slice; compute std on GPU.
            with torch.no_grad():
                final_actions_t = chains_trajs_gpu[:, :, -1, : self.act_steps]
                rollout_mean_action_std = float(
                    final_actions_t.std(dim=(0, 1, 2)).mean()
                )

            # Update aggregates (mean-over-all-batches, avoids last-batch bias).
            # Single GPU->CPU sync here consolidates all per-batch stats.
            if _agg_count > 0:
                n = _agg_count
                with torch.no_grad():
                    stacked = torch.stack(
                        [
                            _agg_t["loss"], _agg_t["pg"], _agg_t["v"], _agg_t["bc"],
                            _agg_t["eta_"], _agg_t["kl"], _agg_t["kl_max"],
                            _agg_t["ratio"], _agg_t["actor_gn"], _agg_t["critic_gn"],
                            _agg_t["clipfrac"],
                        ]
                    ).detach().cpu().numpy()
                loss_mean = float(stacked[0] / n)
                pg_mean = float(stacked[1] / n)
                value_mean = float(stacked[2] / n)
                bc_mean = float(stacked[3] / n)
                eta_mean = float(stacked[4] / n)
                kl_mean = float(stacked[5] / n)
                kl_max = float(stacked[6])
                ratio_mean = float(stacked[7] / n)
                actor_gn_mean = float(stacked[8] / n)
                critic_gn_mean = float(stacked[9] / n)
                clipfrac_mean = float(stacked[10] / n)
            else:
                loss_mean = pg_mean = value_mean = bc_mean = eta_mean = 0.0
                kl_mean = kl_max = ratio_mean = 0.0
                actor_gn_mean = critic_gn_mean = 0.0
                clipfrac_mean = 0.0

            # GAE diagnostics (only valid on train iters) -- compute on GPU.
            if not eval_mode:
                with torch.no_grad():
                    diag = torch.stack(
                        [
                            advantages_trajs_t.mean(),
                            advantages_trajs_t.std(unbiased=False),
                            returns_trajs_t.mean(),
                            values_trajs_t.mean(),
                            values_trajs_t.std(unbiased=False),
                        ]
                    ).detach().cpu().numpy()
                adv_mean = float(diag[0])
                adv_std = float(diag[1])
                ret_mean = float(diag[2])
                val_mean = float(diag[3])
                val_std = float(diag[4])
            else:
                adv_mean = adv_std = ret_mean = val_mean = val_std = 0.0

            # System stats.
            if torch.cuda.is_available():
                gpu_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
                gpu_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
                torch.cuda.reset_peak_memory_stats()
            else:
                gpu_allocated_gb = gpu_reserved_gb = 0.0

            iter_env_steps = self.n_steps * self.n_envs * self.act_steps
            total_fps = iter_env_steps / iter_time if iter_time > 0 else 0.0

            if self.itr % self.log_freq == 0:
                iter_duration = timer()
                run_results[-1]["time"] = iter_duration
                if self._boxed_summary:
                    summary = _format_iter_summary(
                        it=self.itr,
                        total_it=self.n_train_itr,
                        start_it=self._start_itr,
                        tot_time=self._tot_wall_time,
                        tot_timesteps=self._tot_timesteps,
                        collect_time=collect_time,
                        learn_time=learn_time,
                        n_steps=self.n_steps,
                        n_envs=self.n_envs,
                        act_steps=self.act_steps,
                        eval_mode=eval_mode,
                        loss=loss_mean,
                        pg_loss=pg_mean,
                        v_loss=value_mean,
                        bc_loss=bc_mean,
                        eta=eta_mean,
                        approx_kl=kl_mean,
                        clipfrac=clipfrac_mean,
                        explained_var=float(explained_var) if not np.isnan(explained_var) else 0.0,
                        ratio=ratio_mean,
                        avg_episode_reward=float(avg_episode_reward),
                        avg_best_reward=float(avg_best_reward),
                        success_rate=float(success_rate),
                        num_episode_finished=int(num_episode_finished),
                        mean_episode_length=mean_episode_length,
                        actor_lr=float(self.actor_optimizer.param_groups[0]["lr"]),
                        critic_lr=float(self.critic_optimizer.param_groups[0]["lr"]),
                        diffusion_min_sampling_std=float(diffusion_min_sampling_std),
                    )
                    print(summary, flush=True)
                if eval_mode:
                    if self.use_wandb:
                        eval_log = {
                            "Eval/success_rate": success_rate,
                            "Eval/mean_reward": avg_episode_reward,
                            "Eval/mean_best_reward": avg_best_reward,
                            "Eval/mean_episode_length": mean_episode_length,
                            "Eval/episodes_completed": num_episode_finished,
                            "Perf/collection_time": collect_time,
                            "Perf/iteration_time": iter_time,
                            "Perf/total_fps": total_fps,
                            "Train/total_env_steps": cnt_train_step,
                            "System/gpu_memory_allocated_gb": gpu_allocated_gb,
                            "System/gpu_memory_reserved_gb": gpu_reserved_gb,
                        }
                        eval_log.update(self._collect_iter_videos())
                        wandb.log(eval_log, step=self.itr, commit=False)
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    if self.use_wandb:
                        train_log = {
                            # Rewards + episode stats
                            "Train/mean_reward": avg_episode_reward,
                            "Train/mean_best_reward": avg_best_reward,
                            "Train/success_rate": success_rate,
                            "Train/mean_episode_length": mean_episode_length,
                            "Train/episodes_completed": num_episode_finished,
                            "Train/total_env_steps": cnt_train_step,
                            # PPO update losses (mean over all batches)
                            "Loss/total": loss_mean,
                            "Loss/policy": pg_mean,
                            "Loss/value": value_mean,
                            "Loss/bc": bc_mean,
                            "Loss/approx_kl": kl_mean,
                            "Loss/approx_kl_max": kl_max,
                            "Loss/clipfrac": clipfrac_mean,
                            "Loss/explained_variance": explained_var,
                            "Loss/policy_ratio": ratio_mean,
                            "Loss/actor_grad_norm": actor_gn_mean,
                            "Loss/critic_grad_norm": critic_gn_mean,
                            "Loss/actor_lr": self.actor_optimizer.param_groups[0]["lr"],
                            "Loss/critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                            # Policy behaviour
                            "Policy/diffusion_eta": eta_mean,
                            "Policy/diffusion_min_sampling_std": diffusion_min_sampling_std,
                            # GAE diagnostics
                            "Train/mean_advantage": adv_mean,
                            "Train/std_advantage": adv_std,
                            "Train/mean_return": ret_mean,
                            "Train/mean_value": val_mean,
                            "Train/std_value": val_std,
                            # Rollout diagnostics
                            "Rollout/mean_reward_per_step": rollout_mean_reward_per_step,
                            "Rollout/terminated_rate": rollout_terminated_rate,
                            "Rollout/truncated_rate": rollout_truncated_rate,
                            "Rollout/mean_action_std": rollout_mean_action_std,
                            # Perf
                            "Perf/collection_time": collect_time,
                            "Perf/learning_time": learn_time,
                            "Perf/iteration_time": iter_time,
                            "Perf/total_fps": total_fps,
                            # System
                            "System/gpu_memory_allocated_gb": gpu_allocated_gb,
                            "System/gpu_memory_reserved_gb": gpu_reserved_gb,
                        }
                        train_log.update(self._collect_iter_videos())
                        wandb.log(train_log, step=self.itr, commit=True)
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
