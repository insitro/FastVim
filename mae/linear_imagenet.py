import gc
import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from timm.layers import trunc_normal_
from torchmetrics.functional import accuracy


class SupervisedModule(pl.LightningModule):
    def __init__(
        self,
        backbone,
        num_classes: int,
        weight_decay,
        blr,
        batch_size,
        warmup_epochs=10,
        scheduling_epochs=90,
        min_lr=0,
        log_steps_for_model_debugging: int = 500,
        average="micro",
    ):
        """PyTorch Lightning module.

        Parameters
        ----------
        backbone: nn.Module
            The backbone module.
        num_classes: int
            The total number of classes.
        log_steps_for_model_debugging: int
            How often to log full model outputs.
        """
        super().__init__()
        self.backbone = backbone

        trunc_normal_(self.backbone.head.weight, std=0.01)

        # for linear prob only
        # hack: revise model's head with BN
        self.backbone.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(
                self.backbone.head.in_features, affine=False, eps=1e-6
            ),
            self.backbone.head,
        )
        # freeze all but the head
        for _, p in self.backbone.named_parameters():
            p.requires_grad = False
        for _, p in self.backbone.head.named_parameters():
            p.requires_grad = True

        print(
            "number of params trainable (M): %.2f"
            % (
                sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
                / 1.0e6
            )
        )
        print(
            "number of params frozen (M): %.2f"
            % (
                sum(
                    p.numel() for p in self.backbone.parameters() if not p.requires_grad
                )
                / 1.0e6
            )
        )

        self.num_classes = num_classes
        self.weight_decay = weight_decay

        self.average = average

        self.blr = blr
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.scheduling_epochs = scheduling_epochs
        self.min_lr = min_lr

        print("base lr: %.2e" % self.blr)

        self.loss = torch.nn.CrossEntropyLoss()

        self.val_loss = torch.nn.CrossEntropyLoss()
        self.log_steps_for_model_debugging = log_steps_for_model_debugging
        self.save_hyperparameters(ignore=["backbone"])

    def on_train_start(self) -> None:
        print(
            "len(self.trainer.train_dataloader): ", len(self.trainer.train_dataloader)
        )

        """Perform some sanity checks on the dataloader."""
        assert self.trainer.max_epochs is not None
        print("self.trainer.num_training_batches", self.trainer.num_training_batches)
        print("self.trainer.max_epochs", self.trainer.max_epochs)

    def forward(
        self, imgs: torch.Tensor, labels: torch.Tensor, training: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # print('labels.shape===========', labels.shape)

        logit = self.backbone(imgs)
        # print(logit.shape, imgs.shape, labels.shape)
        return logit, labels

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # We use manual optimization here
        # get the optimizer and set the lr / wd rate
        optimizer = self.optimizers(use_pl_optimizer=True)
        assert isinstance(optimizer, LightningOptimizer)

        epoch = self.current_epoch + (batch_idx / len(self.trainer.train_dataloader))

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

        imgs, labels = batch
        logit, labels = self.forward(imgs, labels, training=True)
        loss = self.loss(logit, labels)

        self.log("learning_rate", lr)
        self.log("train_loss", loss, on_step=True, prog_bar=True, rank_zero_only=True)

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # We use manual optimization here
        # get the optimizer and set the lr / wd rate

        imgs, labels = batch

        logit, labels = self.forward(imgs, labels)

        loss = self.val_loss(logit, labels)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            prog_bar=True,
            rank_zero_only=False,
            sync_dist=True,
        )

        acc = accuracy(
            torch.softmax(logit, dim=1),
            labels,
            num_classes=self.num_classes,
            task="multiclass",
            average=self.average,
        )
        self.log(
            "val_acc",
            acc,
            on_step=False,
            prog_bar=True,
            rank_zero_only=False,
            sync_dist=True,
        )

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # We use manual optimization here
        # get the optimizer and set the lr / wd rate

        imgs, labels = batch
        logit, labels = self.forward(imgs, labels)
        loss = self.val_loss(logit, labels)

        self.log("test_loss", loss, prog_bar=True, rank_zero_only=False, sync_dist=True)

        acc = accuracy(
            torch.softmax(logit, dim=1),
            labels,
            num_classes=self.num_classes,
            task="multiclass",
            average=self.average,
        )
        self.log("test_acc", acc, prog_bar=True, rank_zero_only=False, sync_dist=True)

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        imgs, labels = batch
        return self.backbone(imgs)

    def on_train_epoch_start(self):
        print("Starting a new epoch!")

    def on_train_epoch_end(self):
        # Clear CUDA cache
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()

    def configure_optimizers(self) -> torch.optim.AdamW:
        eff_batch_size = self.batch_size * self.trainer.world_size
        self.lr = self.blr * eff_batch_size / 256.0

        print("trainer.accumulate_grad_batches: ", self.trainer.accumulate_grad_batches)
        print("effective batch size: %d" % eff_batch_size)
        print("self.lr: ", self.lr)
        print("self.trainer.world_size: ", self.trainer.world_size)

        """Loading optimizer and learning rate / weight decay schedulers"""
        # optimizer = LARS(self.backbone.head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(
            self.backbone.head.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=0.9,
        )

        print(optimizer)

        return optimizer
