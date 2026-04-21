"""
Parent pre-training agent class.

"""

from __future__ import annotations

import os
import random
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import logging
import wandb
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

log = logging.getLogger(__name__)
from util.scheduler import CosineAnnealingWarmupRestarts

DEVICE = "cuda:0"


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch, device="cuda:0"):
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)


class EMA:
    """
    Empirical moving average

    """

    def __init__(self, cfg):
        super().__init__()
        self.beta = cfg.decay

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class PreTrainAgent:

    def __init__(self, cfg):
        super().__init__()
        self.seed = cfg.get("seed", 42)
        self.device = cfg.device
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Wandb
        self.use_wandb = cfg.wandb is not None
        if cfg.wandb is not None:
            self._install_safe_wandb_logging()
            try:
                wandb.init(
                    entity=cfg.wandb.entity,
                    project=cfg.wandb.project,
                    name=cfg.wandb.run,
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
            except Exception:
                self.use_wandb = False
                log.exception("wandb.init failed; disabling W&B logging for this run.")
        self._log_model_artifacts = (
            bool(cfg.wandb.get("log_model", True)) if cfg.wandb is not None else False
        )

        # Build model
        self.model = hydra.utils.instantiate(cfg.model)
        self.ema = EMA(cfg.ema)
        self.ema_model = deepcopy(self.model)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.learning_rate,
            min_lr=cfg.train.lr_scheduler.min_lr,
            warmup_steps=cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.reset_parameters()

        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.epoch_start_ema = cfg.train.get("epoch_start_ema", 20)
        self.update_ema_freq = cfg.train.get("update_ema_freq", 10)
        self.val_freq = cfg.train.get("val_freq", 100)
        cpu_num_workers = cfg.train.get("num_workers", 4)

        # Logging, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq
        self.checkpoint_eval_cfg = cfg.train.get("checkpoint_eval", None)

        # Async checkpoint eval -- lazy init (deferred to first save_model
        # so we pass the current EMA state_dict into the worker).
        self._async_eval_enabled = self._resolve_async_eval_enabled()
        self._async_eval_manager = None
        self._async_rollout_best: tuple[float, float, float] | None = None
        self._async_rollout_best_epoch: int | None = None
        if self._async_eval_enabled and self.use_wandb:
            # Async results log at past epochs; use local_step to avoid
            # wandb's backward-step rejection.
            try:
                wandb.define_metric("local_step")
                wandb.define_metric("*", step_metric="local_step")
            except Exception:
                log.exception(
                    "wandb.define_metric(local_step) failed; async eval "
                    "results may plot on wandb's implicit step."
                )

        # Build dataset
        self.dataset_train = hydra.utils.instantiate(cfg.train_dataset)
        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=cpu_num_workers if self.dataset_train.device == "cpu" else 0,
            shuffle=True,
            pin_memory=True if self.dataset_train.device == "cpu" else False,
        )
        self.dataloader_val = None
        if "train_split" in cfg.train and cfg.train.train_split < 1:
            val_indices = self.dataset_train.set_train_val_split(cfg.train.train_split)
            self.dataset_val = deepcopy(self.dataset_train)
            self.dataset_val.set_indices(val_indices)
            self.dataloader_val = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=cpu_num_workers if self.dataset_val.device == "cpu" else 0,
                shuffle=True,
                pin_memory=True if self.dataset_val.device == "cpu" else False,
            )

    def _install_safe_wandb_logging(self) -> None:
        if hasattr(wandb, "_rl_mimicgen_safe_log_installed"):
            return

        wandb._rl_mimicgen_original_log = wandb.log

        def _safe_log(*args, **kwargs):
            try:
                return wandb._rl_mimicgen_original_log(*args, **kwargs)
            except Exception:
                self.use_wandb = False
                log.exception("wandb.log failed; disabling W&B logging for the rest of this run.")
                return None

        wandb.log = _safe_log
        wandb._rl_mimicgen_safe_log_installed = True

    def run(self):
        raise NotImplementedError

    # Async checkpoint-eval integration

    def _resolve_async_eval_enabled(self) -> bool:
        eval_cfg = getattr(self, "checkpoint_eval_cfg", None)
        if eval_cfg is None:
            return False
        if not eval_cfg.get("enabled", False):
            return False
        return bool(eval_cfg.get("async_enabled", False))

    def _wandb_log(self, payload: dict[str, object], epoch: int, commit: bool = True) -> None:
        """Wandb ``log`` helper that plays nicely with async past-epoch posts.

        When async eval is on we rely on ``define_metric(step_metric=
        'local_step')`` set at init, so all metrics get ``local_step``
        added to the payload and ``step=`` is dropped (wandb disallows
        going backwards in its internal step). When async is off we keep
        upstream's existing ``step=epoch`` behavior bit-for-bit.
        """
        if not self.use_wandb:
            return
        try:
            if self._async_eval_enabled:
                payload = dict(payload)
                payload["local_step"] = int(epoch)
                wandb.log(payload, commit=commit)
            else:
                wandb.log(payload, step=int(epoch), commit=commit)
        except Exception:
            log.exception("wandb.log failed: %s", list(payload))

    def _ensure_async_eval_manager(self) -> None:
        if not self._async_eval_enabled or self._async_eval_manager is not None:
            return
        eval_cfg = self.checkpoint_eval_cfg
        # Lazy import -- avoids pulling hydra/rl_mimicgen on non-async runs.
        try:
            from rl_mimicgen.dppo_async import AsyncCheckpointEvalManager
        except Exception:
            log.exception(
                "Failed to import AsyncCheckpointEvalManager; "
                "falling back to the subprocess eval path."
            )
            self._async_eval_enabled = False
            return

        video_dir = None
        video_mode = str(eval_cfg.get("video_checkpoints", "none"))
        save_video = video_mode != "none"
        if save_video:
            output_dir = eval_cfg.get("output_dir", None)
            if output_dir is None:
                output_dir = os.path.join(self.logdir, "checkpoint_eval")
            video_dir = os.path.join(str(output_dir), "videos")

        try:
            self._async_eval_manager = AsyncCheckpointEvalManager(
                eval_config_dir=eval_cfg.config_dir,
                eval_config_name=eval_cfg.config_name,
                initial_ema_state_dict=self.ema_model.state_dict(),
                device=str(eval_cfg.get("device", self.device)),
                video_dir=video_dir,
                save_video=save_video,
                render_num=eval_cfg.get("render_num", None),
                n_envs=eval_cfg.get("n_envs", None),
                n_steps=eval_cfg.get("n_steps", None),
                n_episodes=eval_cfg.get("n_episodes", None),
                max_episode_steps=eval_cfg.get("max_episode_steps", None),
                queue_size=int(eval_cfg.get("async_queue_size", 2)),
            )
        except Exception:
            log.exception(
                "AsyncCheckpointEvalManager init failed; disabling "
                "async eval for this run (subprocess path will be used)."
            )
            self._async_eval_manager = None
            self._async_eval_enabled = False

    def _consume_async_eval_results(self, results: list[AsyncEvalResult]) -> None:
        """Log + best-track every completed async eval result."""
        for result in results:
            if result.error is not None:
                log.error(
                    "Async eval for epoch %d failed: %s",
                    result.epoch,
                    result.error,
                )
                continue
            metrics = result.metrics
            log.info(
                "Async eval @ epoch %d: success %.4f | return %.4f | "
                "best %.4f | wall %.2fs",
                result.epoch,
                float(metrics.get("success_rate", 0.0)),
                float(metrics.get("return_mean", 0.0)),
                float(metrics.get("best_reward_mean", 0.0)),
                float(result.wall_time),
            )
            self._wandb_log(
                {
                    "Eval/success_rate": float(metrics.get("success_rate", 0.0)),
                    "Eval/mean_reward": float(metrics.get("return_mean", 0.0)),
                    "Eval/mean_best_reward": float(
                        metrics.get("best_reward_mean", 0.0)
                    ),
                    "Eval/episodes_completed": int(metrics.get("num_episode", 0)),
                    "Perf/eval_wall_time": float(result.wall_time),
                },
                epoch=result.epoch,
            )
            # Single ``video`` panel (``video_k`` when >1 render env) matches
            # the finetune wandb layout.
            valid_video_paths = [
                (idx, vp)
                for idx, vp in enumerate(result.video_paths)
                if vp and os.path.isfile(vp)
            ]
            video_payload: dict = {}
            for idx, video_path in valid_video_paths:
                key = "video" if len(valid_video_paths) == 1 else f"video_{idx}"
                try:
                    video_payload[key] = wandb.Video(
                        video_path, fps=20, format="mp4"
                    )
                except Exception:
                    log.exception(
                        "wandb video wrap failed for %s", video_path
                    )
            if video_payload:
                try:
                    self._wandb_log(video_payload, epoch=result.epoch)
                except Exception:
                    log.exception("wandb video log failed")
            self._maybe_save_best_async_result(result)

    def _maybe_save_best_async_result(self, result: AsyncEvalResult) -> None:
        """Save the exact-weight snapshot if this result improves the best."""
        if result.ema_state_dict is None:
            return
        eval_cfg = self.checkpoint_eval_cfg
        copy_best_to = (
            eval_cfg.get("copy_best_to", None) if eval_cfg is not None else None
        )
        if copy_best_to is None:
            return
        metrics = result.metrics
        rank = (
            float(metrics.get("success_rate", 0.0)),
            float(metrics.get("best_reward_mean", 0.0)),
            float(metrics.get("return_mean", 0.0)),
        )
        if self._async_rollout_best is not None and rank <= self._async_rollout_best:
            return
        self._async_rollout_best = rank
        self._async_rollout_best_epoch = int(result.epoch)
        copy_best_to_path = Path(str(copy_best_to)).expanduser()
        copy_best_to_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": int(result.epoch),
                "ema": result.ema_state_dict,
                "eval_metrics": metrics,
            },
            copy_best_to_path,
        )
        log.info(
            "Saved best-so-far async-eval checkpoint (epoch=%d, success=%.4f) to %s",
            int(result.epoch),
            rank[0],
            copy_best_to_path,
        )
        self._log_checkpoint_artifact(
            copy_best_to_path, int(result.epoch), extra_aliases=["best"]
        )

    def drain_async_eval(self) -> None:
        if self._async_eval_manager is None:
            return
        self._consume_async_eval_results(self._async_eval_manager.drain())

    def shutdown_async_eval(self, timeout: float | None = None) -> None:
        if self._async_eval_manager is None:
            return
        remaining = self._async_eval_manager.drain_blocking(timeout=timeout)
        self._consume_async_eval_results(remaining)
        self._async_eval_manager.close()
        self._async_eval_manager = None

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.epoch < self.epoch_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save_model(self):
        """
        saves model and ema to disk;
        """
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")
        self._log_checkpoint_artifact(savepath, self.epoch)
        self._maybe_run_checkpoint_eval()
        return savepath

    def _log_checkpoint_artifact(
        self,
        path: str | os.PathLike[str],
        step: int,
        extra_aliases: list[str] | None = None,
    ) -> None:
        """Upload a checkpoint file to wandb as a versioned artifact.

        Name is ``<run_id>-checkpoint``; each call publishes a new version
        with aliases ``step-<step>`` + ``latest`` (plus any extras). No-op
        if wandb is off or ``cfg.wandb.log_model`` is False.
        """
        if not self.use_wandb or not self._log_model_artifacts:
            return
        run = wandb.run
        if run is None:
            return
        try:
            artifact = wandb.Artifact(
                name=f"{run.id}-checkpoint",
                type="model",
                metadata={"step": int(step)},
            )
            artifact.add_file(str(path))
            aliases = [f"step-{int(step)}", "latest"]
            if extra_aliases:
                aliases.extend(extra_aliases)
            wandb.log_artifact(artifact, aliases=aliases)
        except Exception:
            log.exception("wandb.log_artifact failed for %s", path)

    def _maybe_run_checkpoint_eval(self) -> None:
        eval_cfg = self.checkpoint_eval_cfg
        if eval_cfg is None or not eval_cfg.get("enabled", False):
            return

        # Async path: submit EMA weights to background manager; subprocess fallback below.
        if self._async_eval_enabled:
            self._ensure_async_eval_manager()
            if self._async_eval_manager is not None:
                self._async_eval_manager.submit(
                    self.epoch, self.ema_model.state_dict()
                )
                return

        repo_root = Path(__file__).resolve().parents[2]
        script_path = Path(
            eval_cfg.get(
                "script_path",
                repo_root / "script" / "eval_checkpoint_sweep.py",
            )
        ).expanduser()
        command = [
            sys.executable,
            str(script_path),
            "--config-dir",
            str(Path(eval_cfg.config_dir).expanduser()),
            "--config-name",
            str(eval_cfg.config_name),
            "--checkpoint-dir",
            str(Path(self.checkpoint_dir).expanduser()),
            "--output-dir",
            str(Path(eval_cfg.output_dir).expanduser()),
            "--device",
            str(eval_cfg.get("device", self.device)),
            "--every-n",
            str(eval_cfg.get("every_n", 1)),
            "--video-checkpoints",
            str(eval_cfg.get("video_checkpoints", "none")),
            "--render-num",
            str(eval_cfg.get("render_num", 1)),
        ]
        if "n_envs" in eval_cfg and eval_cfg.n_envs is not None:
            command.extend(["--n-envs", str(eval_cfg.n_envs)])
        if "n_steps" in eval_cfg and eval_cfg.n_steps is not None:
            command.extend(["--n-steps", str(eval_cfg.n_steps)])
        if "max_episode_steps" in eval_cfg and eval_cfg.max_episode_steps is not None:
            command.extend(["--max-episode-steps", str(eval_cfg.max_episode_steps)])
        if "start_index" in eval_cfg and eval_cfg.start_index is not None:
            command.extend(["--start-index", str(eval_cfg.start_index)])
        if "end_index" in eval_cfg and eval_cfg.end_index is not None:
            command.extend(["--end-index", str(eval_cfg.end_index)])
        if eval_cfg.get("skip_existing", True):
            command.append("--skip-existing")
        if "copy_best_to" in eval_cfg and eval_cfg.copy_best_to is not None:
            command.extend(["--copy-best-to", str(Path(eval_cfg.copy_best_to).expanduser())])

        log.info("Running checkpoint evaluation sweep: %s", " ".join(command))
        subprocess.run(command, cwd=repo_root, check=True)

    def load(self, epoch):
        """
        loads model and ema from disk
        """
        loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")
        data = torch.load(loadpath, weights_only=True)

        self.epoch = data["epoch"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])
