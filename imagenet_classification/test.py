from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
from utils import _maybe_instantiate, load_config

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_name", default="FastVimT.yaml")
    parser.add_argument("--precision", default=32)  # 16-mixed
    parser.add_argument(
        "--ckpt_path", default="ckpt_path"
    )  # replace with ImageNet supervised checkpoint. Model wll return both ema and non-ema results.

    args = parser.parse_args()

    full_config = load_config(Path(__file__).parent / "config", args.config_name)
    pl.seed_everything(full_config.pl_seed, workers=True)

    trainer = pl.Trainer(
        logger=None,
        num_nodes=full_config.num_nodes,
        precision=args.precision,
        sync_batchnorm=True,
    )

    data_loader: Union[torch.utils.data.DataLoader, pl.LightningDataModule] = (
        _maybe_instantiate(full_config.data_config)
    )

    model: pl.LightningModule = _maybe_instantiate(full_config.model_config)

    trainer.test(model, dataloaders=data_loader, ckpt_path=args.ckpt_path)
