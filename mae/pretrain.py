from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import _maybe_instantiate, load_config

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_name", default="pretrain_FastVimB.yaml")
    parser.add_argument("--model_save_dir", default="checkpoints_mae_pretrain_IN1k/")
    parser.add_argument("--checkpoint_every_n_epochs", default=400)
    parser.add_argument("--log_steps", default=5)
    parser.add_argument("--precision", default=32)  # 16-mixed

    args = parser.parse_args()

    full_config = load_config(Path(__file__).parent / "config", args.config_name)
    pl.seed_everything(full_config.pl_seed, workers=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_save_dir + args.config_name.split(".")[0] + "/",
        filename="epoch={epoch:02d}-train_loss={train_loss:.3f}",
        monitor="train_loss",
        save_top_k=-1,  # Save all checkpoints
        save_weights_only=False,
        enable_version_counter=True,
        every_n_epochs=args.checkpoint_every_n_epochs,  # Save checkpoint every n epochs
    )

    print("accum_iter: ", full_config.accum_iter)

    trainer = pl.Trainer(
        logger=None,
        num_nodes=full_config.num_nodes,
        max_epochs=full_config.training_epochs,
        log_every_n_steps=args.log_steps,
        precision=args.precision,
        callbacks=[checkpoint_callback],
        sync_batchnorm=True,
        accumulate_grad_batches=full_config.accum_iter,
    )

    data_loader: Union[torch.utils.data.DataLoader, pl.LightningDataModule] = (
        _maybe_instantiate(full_config.data_config)
    )

    model: pl.LightningModule = _maybe_instantiate(full_config.model_config)

    trainer.fit(model, train_dataloaders=data_loader, ckpt_path=None)
