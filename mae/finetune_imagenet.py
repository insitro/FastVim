import math

import lr_decay as lrd
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torchmetrics.functional import accuracy


class SupervisedModule(pl.LightningModule):
    def __init__(
        self,
        backbone,
        num_classes: int,
        mixup,
        cutmix,
        mixup_mode,
        mixup_prob,
        mixup_switch_prob,
        weight_decay,
        blr,
        batch_size,
        warmup_epochs,
        scheduling_epochs,
        accum_iter,
        min_lr,
        layer_decay,
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
        num_classes: int
            The total number of classes.
        label_smoothing: float
            How much label smoothing to apply.
        log_steps_for_model_debugging: int
            How often to log full model outputs.
        """
        super().__init__()
        self.backbone = backbone

        self.num_classes = num_classes
        self.weight_decay = weight_decay

        self.average = average
        self.accum_iter = accum_iter

        self.blr = blr
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.scheduling_epochs = scheduling_epochs
        self.min_lr = min_lr
        self.layer_decay = layer_decay

        print("base lr: %.2e" % self.blr)

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
        print(
            "len(self.trainer.train_dataloader): ", len(self.trainer.train_dataloader)
        )

        """Perform some sanity checks on the dataloader."""
        assert self.trainer.max_epochs is not None
        print("self.trainer.num_training_batches", self.trainer.num_training_batches)
        print("self.trainer.max_epochs", self.trainer.max_epochs)

        # # Clear CUDA cache
        # torch.cuda.empty_cache()
        # # Force garbage collection
        # gc.collect()

    def forward(
        self, imgs: torch.Tensor, labels: torch.Tensor, training: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # print('labels.shape===========', labels.shape)
        if training:
            if self.mixup_fn is not None:
                imgs, labels = self.mixup_fn(imgs, labels)

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

        imgs, labels = batch
        logit, labels = self.forward(imgs, labels, training=True)
        loss = self.loss(logit, labels)

        if batch_idx % self.accum_iter == 0:
            self.log("learning_rate", lr)
            self.log(
                "train_loss", loss, on_step=True, prog_bar=True, rank_zero_only=True
            )

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

    # def on_train_epoch_end(self):
    #     # Clear CUDA cache
    #     torch.cuda.empty_cache()
    #     # Force garbage collection
    #     gc.collect()

    def configure_optimizers(self) -> torch.optim.AdamW:
        eff_batch_size = self.batch_size * self.accum_iter * self.trainer.world_size
        self.lr = self.blr * eff_batch_size / 256.0

        print("trainer.accumulate_grad_batches: ", self.trainer.accumulate_grad_batches)
        print("effective batch size: %d" % eff_batch_size)
        print("self.lr: ", self.lr)
        print("self.trainer.world_size: ", self.trainer.world_size)
        print("accumulate grad iterations: %d" % self.accum_iter)

        """Loading optimizer and learning rate / weight decay schedulers"""
        # param_groups = optim_factory.add_weight_decay(self.backbone, self.weight_decay)

        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(
            self.backbone,
            self.weight_decay,
            no_weight_decay_list=self.backbone.no_weight_decay(),
            layer_decay=self.layer_decay,
        )

        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.999))
        print(optimizer)

        return optimizer
