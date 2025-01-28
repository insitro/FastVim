import gc

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torchmetrics.functional import accuracy
from utils import get_params_groups_wd


class SupervisedModule(pl.LightningModule):
    def __init__(
        self,
        backbone,
        lr_schedule: np.ndarray,
        wd_schedule: np.ndarray,
        num_classes: int,
        mixup,
        cutmix,
        mixup_mode,
        mixup_prob,
        mixup_switch_prob,
        cutmix_minmax=None,
        label_smoothing: float = 0.0,
        log_steps_for_model_debugging: int = 500,
        average="micro",
    ):
        """PyTorch Lightning module.

        Parameters
        ----------
        backbone: nn.Module
            The backbone module.
        lr_schedule: np.ndarray
            An array containing the learning rate at each epoch.
        wd_schedule: np.ndarray
            An array containing the weight decay parameters at each epoch.
        num_classes: int
            The total number of classes.
        label_smoothing: float
            How much label smoothing to apply.
        log_steps_for_model_debugging: int
            How often to log full model outputs.
        """
        super().__init__()
        self.backbone = backbone

        self.average = average

        self.num_classes = num_classes

        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule

        self.mixup_fn = None
        mixup_active = mixup > 0 or cutmix > 0.0 or cutmix_minmax is not None
        if mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=mixup,
                cutmix_alpha=cutmix,
                cutmix_minmax=cutmix_minmax,
                prob=mixup_prob,
                switch_prob=mixup_switch_prob,
                mode=mixup_mode,
                label_smoothing=label_smoothing,
                num_classes=num_classes,
            )

        if mixup_active:
            # smoothing is handled with mixup label transform
            print("using mixup_active --------------")
            self.loss = SoftTargetCrossEntropy()

        elif label_smoothing > 0:
            print("using LabelSmoothingCrossEntropy --------------", label_smoothing)
            self.loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            print("using CrossEntropyLoss --------------")
            self.loss = torch.nn.CrossEntropyLoss()

        self.val_loss = torch.nn.CrossEntropyLoss()
        self.log_steps_for_model_debugging = log_steps_for_model_debugging
        self.save_hyperparameters(ignore=["backbone"])

    def on_train_start(self) -> None:
        """Perform some sanity checks on the dataloader."""
        assert self.trainer.max_epochs is not None
        train_steps = self.trainer.num_training_batches * self.trainer.max_epochs

        print("train_steps, len(self.lr_schedule)", train_steps, len(self.lr_schedule))
        print("self.trainer.num_training_batches", self.trainer.num_training_batches)
        print("self.trainer.max_epochs", self.trainer.max_epochs)

        assert train_steps <= len(self.lr_schedule)
        assert train_steps <= len(self.wd_schedule)

    def forward(
        self, imgs: torch.Tensor, labels: torch.Tensor, training: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if training:
            if self.mixup_fn is not None:
                imgs, labels = self.mixup_fn(imgs, labels)

        logit = self.backbone(imgs)

        return logit, labels

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # We use manual optimization here
        # get the optimizer and set the lr / wd rate
        optimizer = self.optimizers(use_pl_optimizer=True)
        assert isinstance(optimizer, LightningOptimizer)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[self.global_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[self.global_step]

        imgs, labels = batch
        logit, labels = self.forward(imgs, labels, training=True)
        loss = self.loss(logit, labels)

        self.log("learning_rate", self.lr_schedule[self.global_step])
        self.log("train_loss", loss, on_step=True, prog_bar=True, rank_zero_only=True)
        self.log("weight_decay", self.wd_schedule[self.global_step])

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

    def on_train_epoch_end(self):
        # Clear CUDA cache
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()

    def configure_optimizers(self) -> torch.optim.AdamW:
        """Loading optimizer and learning rate / weight decay schedulers"""

        params_groups = get_params_groups_wd(self.backbone)

        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        return optimizer
