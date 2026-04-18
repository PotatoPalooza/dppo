"""
Parent pre-training agent class.

"""

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
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Build model
        self.model = hydra.utils.instantiate(cfg.model)
        self.ema = EMA(cfg.ema)
        self.ema_model = deepcopy(self.model)

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

    def run(self):
        raise NotImplementedError

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
        self._maybe_run_checkpoint_eval()
        return savepath

    def _maybe_run_checkpoint_eval(self):
        eval_cfg = self.checkpoint_eval_cfg
        if eval_cfg is None or not eval_cfg.get("enabled", False):
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
