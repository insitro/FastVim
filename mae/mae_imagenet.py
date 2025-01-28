import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer


class SSLModule(pl.LightningModule):
    def __init__(
        self,
        backbone,
        weight_decay,
        blr,
        batch_size,
        warmup_epochs,
        scheduling_epochs,
        accum_iter,
        min_lr,
        mask_ratio,
    ):
        """PyTorch Lightning module.

        Parameters
        ----------
        backbone: nn.Module
            The backbone module.
        """
        super().__init__()
        self.backbone = backbone
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["backbone"])

        self.blr = blr
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.scheduling_epochs = scheduling_epochs
        self.accum_iter = accum_iter
        self.mask_ratio = mask_ratio
        self.min_lr = min_lr

        print("base lr: %.2e" % self.blr)

    def on_train_start(self) -> None:
        print(
            "len(self.trainer.train_dataloader): ", len(self.trainer.train_dataloader)
        )

        """Perform some sanity checks on the dataloader."""
        assert self.trainer.max_epochs is not None
        print("self.trainer.num_training_batches", self.trainer.num_training_batches)
        print("self.trainer.max_epochs", self.trainer.max_epochs)

    def forward(
        self, imgs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, pred, mask = self.backbone(imgs, self.mask_ratio)
        return loss, pred, mask

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # We use manual optimization here
        # get the optimizer and set the lr / wd rate
        optimizer = self.optimizers(use_pl_optimizer=True)
        assert isinstance(optimizer, LightningOptimizer)

        if batch_idx % self.accum_iter == 0:
            epoch = self.current_epoch + (
                batch_idx / len(self.trainer.train_dataloader)
            )

            if epoch < self.warmup_epochs:
                lr = self.lr * epoch / self.warmup_epochs
            else:
                lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                    1.0
                    + math.cos(
                        math.pi
                        * (epoch - self.warmup_epochs)
                        / (self.scheduling_epochs - self.warmup_epochs)
                    )
                )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                if "lr_scale" in param_group:
                    param_group["lr"] *= param_group["lr_scale"]

        imgs, _ = batch

        loss, pred, mask = self.forward(imgs)

        if batch_idx % self.accum_iter == 0:
            self.log("learning_rate", lr)
            self.log(
                "train_loss", loss, on_step=True, prog_bar=True, rank_zero_only=True
            )

        return loss

    def on_train_epoch_start(self):
        print("Starting a new epoch!")

    def configure_optimizers(self) -> torch.optim.AdamW:
        eff_batch_size = self.batch_size * self.accum_iter * self.trainer.world_size
        self.lr = self.blr * eff_batch_size / 256.0

        print("trainer.accumulate_grad_batches: ", self.trainer.accumulate_grad_batches)
        print("effective batch size: %d" % eff_batch_size)
        print("self.lr: ", self.lr)
        print("self.trainer.world_size: ", self.trainer.world_size)
        print("accumulate grad iterations: %d" % self.accum_iter)

        """Loading optimizer and learning rate / weight decay schedulers"""
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )

        return optimizer
