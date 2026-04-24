"""
Parent PPO fine-tuning agent class.

"""

from typing import Optional
import numpy as np
import torch
import logging
from util.scheduler import CosineAnnealingWarmupRestarts

log = logging.getLogger(__name__)
from agent.finetune.train_agent import TrainAgent
from util.reward_scaling import RunningRewardScaler


class TrainPPOAgent(TrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Batch size for logprobs calculations after an iteration --- prevent out of memory if using a single batch
        self.logprob_batch_size = cfg.train.get("logprob_batch_size", 10000)
        assert (
            self.logprob_batch_size % self.n_envs == 0
        ), "logprob_batch_size must be divisible by n_envs"

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Wwarm up period for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # Optimizer
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        # use cosine scheduler with linear warmup
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        # Generalized advantage estimation
        self.gae_lambda: float = cfg.train.get("gae_lambda", 0.95)

        # If specified, stop gradient update once KL difference reaches it
        self.target_kl: Optional[float] = cfg.train.target_kl

        # Number of times the collected data is used in gradient update
        self.update_epochs: int = cfg.train.update_epochs

        # Entropy loss coefficient
        self.ent_coef: float = cfg.train.get("ent_coef", 0)

        # Value loss coefficient
        self.vf_coef: float = cfg.train.get("vf_coef", 0)

        # Whether to use running reward scaling
        self.reward_scale_running: bool = cfg.train.reward_scale_running
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(self.n_envs)

        # Scaling reward with constant
        self.reward_scale_const: float = cfg.train.get("reward_scale_const", 1)

        # Use base policy
        self.use_bc_loss: bool = cfg.train.get("use_bc_loss", False)
        self.bc_loss_coeff: float = cfg.train.get("bc_loss_coeff", 0)

    def reset_actor_optimizer(self):
        """Not used anywhere currently"""
        new_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=self.cfg.train.actor_lr,
            weight_decay=self.cfg.train.actor_weight_decay,
        )
        new_optimizer.load_state_dict(self.actor_optimizer.state_dict())
        self.actor_optimizer = new_optimizer

        new_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=self.cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=self.cfg.train.actor_lr,
            min_lr=self.cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=self.cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        new_scheduler.load_state_dict(self.actor_lr_scheduler.state_dict())
        self.actor_lr_scheduler = new_scheduler
        log.info("Reset actor optimizer")

    def _checkpoint_state(self):
        payload = {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
            "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
        }
        if self.reward_scale_running:
            payload["running_reward_scaler"] = {
                "ret": self.running_reward_scaler.ret.tolist(),
                "ret_rms_mean": self.running_reward_scaler.ret_rms.mean.tolist(),
                "ret_rms_var": self.running_reward_scaler.ret_rms.var.tolist(),
                "ret_rms_count": float(self.running_reward_scaler.ret_rms.count),
            }
        return payload

    def _restore_checkpoint_state(self, payload):
        actor_optimizer = payload.get("actor_optimizer")
        critic_optimizer = payload.get("critic_optimizer")
        actor_lr_scheduler = payload.get("actor_lr_scheduler")
        critic_lr_scheduler = payload.get("critic_lr_scheduler")
        if actor_optimizer is not None:
            self.actor_optimizer.load_state_dict(actor_optimizer)
        if critic_optimizer is not None:
            self.critic_optimizer.load_state_dict(critic_optimizer)
        if actor_lr_scheduler is not None:
            self.actor_lr_scheduler.load_state_dict(actor_lr_scheduler)
        if critic_lr_scheduler is not None:
            self.critic_lr_scheduler.load_state_dict(critic_lr_scheduler)
        scaler_state = payload.get("running_reward_scaler")
        if self.reward_scale_running and isinstance(scaler_state, dict):
            self.running_reward_scaler.ret = np.asarray(scaler_state.get("ret", []), dtype=float)
            self.running_reward_scaler.ret_rms.mean = np.asarray(
                scaler_state.get("ret_rms_mean", []), dtype=float
            )
            self.running_reward_scaler.ret_rms.var = np.asarray(
                scaler_state.get("ret_rms_var", []), dtype=float
            )
            self.running_reward_scaler.ret_rms.count = float(
                scaler_state.get("ret_rms_count", self.running_reward_scaler.ret_rms.count)
            )
