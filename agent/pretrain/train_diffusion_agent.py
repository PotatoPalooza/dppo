"""
Pre-training diffusion policy

"""

import contextlib
import logging
import wandb
import numpy as np
import torch

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_bf16 = cfg.train.get("use_bf16", False)
        if cfg.train.get("use_compile", False):
            self.model = torch.compile(self.model)
            self.ema_model = torch.compile(self.ema_model)
        if self.resume_path:
            self.load_checkpoint(self.resume_path)

    def _autocast(self):
        if self.use_bf16 and "cuda" in str(self.device):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def run(self):

        timer = Timer()
        while self.epoch <= self.n_epochs:

            # train
            loss_train_epoch = []
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train, self.device)

                self.model.train()
                with self._autocast():
                    loss_train = self.model.loss(*batch_train)
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

                self.optimizer.step()
                self.optimizer.zero_grad()

                # update ema
                if self.cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                self.cnt_batch += 1
            loss_train = np.mean(loss_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    for batch_val in self.dataloader_val:
                        if self.dataset_val.device == "cpu":
                            batch_val = batch_to_device(batch_val, self.device)
                        with self._autocast():
                            loss_val = self.model.loss(*batch_val)
                        loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:
                val_str = f" | val loss {loss_val:8.4f}" if loss_val is not None else ""
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f}{val_str} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    wandb.log(
                        {
                            "loss - train": loss_train,
                        },
                        step=self.epoch,
                        commit=True,
                    )

            # count
            self.epoch += 1
