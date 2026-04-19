"""
Pre-training Gaussian/GMM policy

"""

import logging
import wandb
import numpy as np
import torch

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainGaussianAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Entropy bonus - not used right now since using fixed_std
        self.ent_coef = cfg.train.get("ent_coef", 0)

    def run(self):

        timer = Timer()
        self.epoch = 1
        cnt_batch = 0
        for _ in range(self.n_epochs):

            # train
            loss_train_epoch = []
            ent_train_epoch = []
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train, self.device)

                self.model.train()
                loss_train, infos_train = self.model.loss(
                    *batch_train,
                    ent_coef=self.ent_coef,
                )
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())
                ent_train_epoch.append(infos_train["entropy"].item())

                self.optimizer.step()
                self.optimizer.zero_grad()

                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
            loss_train = np.mean(loss_train_epoch)
            ent_train = np.mean(ent_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val, self.device)
                    loss_val, infos_val = self.model.loss(
                        *batch_val,
                        ent_coef=self.ent_coef,
                    )
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
                epoch_time = timer()  # Timer resets on call → elapsed since last log
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos_train.items()]
                )
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | {infos_str} | t:{epoch_time:8.4f}"
                )
                payload = {
                    "Loss/train": float(loss_train),
                    "Loss/entropy": float(ent_train),
                    "Loss/lr": float(self.optimizer.param_groups[0]["lr"]),
                    "Perf/epoch_time": float(epoch_time),
                    "Train/epoch": int(self.epoch),
                    "Train/batches_seen": int(cnt_batch),
                }
                if loss_val is not None:
                    payload["Loss/val"] = float(loss_val)
                if torch.cuda.is_available() and "cuda" in str(self.device):
                    payload["System/gpu_memory_allocated_gb"] = float(
                        torch.cuda.memory_allocated(self.device) / 1e9
                    )
                    payload["System/gpu_memory_reserved_gb"] = float(
                        torch.cuda.memory_reserved(self.device) / 1e9
                    )
                self._wandb_log(payload, epoch=self.epoch, commit=True)

            # drain any async eval results posted since the last epoch
            self.drain_async_eval()

            # count
            self.epoch += 1

        # training is done — block on any still-in-flight rollouts so
        # their metrics/videos land in wandb before the run finishes.
        self.shutdown_async_eval()
