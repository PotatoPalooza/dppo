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

    def _autocast(self):
        if self.use_bf16 and "cuda" in str(self.device):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def run(self):

        timer = Timer()
        self.epoch = 1
        cnt_batch = 0
        for _ in range(self.n_epochs):

            # train -- accumulate loss tensors on GPU; one sync per epoch
            # instead of per batch. Matters at high batch counts per epoch.
            loss_train_epoch: list[torch.Tensor] = []
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train, self.device)

                self.model.train()
                with self._autocast():
                    loss_train = self.model.loss(*batch_train)
                loss_train.backward()
                loss_train_epoch.append(loss_train.detach())

                self.optimizer.step()
                self.optimizer.zero_grad()

                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
            loss_train = (
                float(torch.stack(loss_train_epoch).mean())
                if loss_train_epoch
                else float("nan")
            )

            # validate
            loss_val_epoch: list[torch.Tensor] = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    for batch_val in self.dataloader_val:
                        if self.dataset_val.device == "cpu":
                            batch_val = batch_to_device(batch_val, self.device)
                        with self._autocast():
                            loss_val = self.model.loss(*batch_val)
                        loss_val_epoch.append(loss_val.detach())
                self.model.train()
            loss_val = (
                float(torch.stack(loss_val_epoch).mean())
                if loss_val_epoch
                else None
            )

            # update lr
            self.lr_scheduler.step()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:
                epoch_time = timer()  # Timer resets on call -> elapsed since last log
                val_str = f" | val loss {loss_val:8.4f}" if loss_val is not None else ""
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f}{val_str} | t:{epoch_time:8.4f}"
                )
                payload = {
                    "Loss/train": loss_train,
                    "Loss/lr": float(self.optimizer.param_groups[0]["lr"]),
                    "Perf/epoch_time": float(epoch_time),
                    "Train/epoch": int(self.epoch),
                    "Train/batches_seen": int(cnt_batch),
                }
                if loss_val is not None:
                    payload["Loss/val"] = loss_val
                if torch.cuda.is_available() and "cuda" in str(self.device):
                    payload["System/gpu_memory_allocated_gb"] = float(
                        torch.cuda.memory_allocated(self.device) / 1e9
                    )
                    payload["System/gpu_memory_reserved_gb"] = float(
                        torch.cuda.memory_reserved(self.device) / 1e9
                    )
                self._wandb_log(payload, epoch=self.epoch, commit=True)

            # Drain async results -- each logs at its own past epoch via local_step.
            self.drain_async_eval()

            # count
            self.epoch += 1

        # training is done -- block on any still-in-flight rollouts so
        # their metrics/videos land in wandb before the run finishes.
        self.shutdown_async_eval()
