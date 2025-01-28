## Model Weights and Configurations

| Model | #param. | Top-1 Acc. | Config | Log |
|:------------------------------------------------------------------:|:-------------:|:----------:|:----------:|:----------:|
| [FastVim-T.ckpt](https://github.com/insitro/FastVim/releases/download/v0/FastVim_tiny.ckpt)    |       7M       |   75.4   | [FastVimT.yaml](config/FastVimT.yaml) | [FastVimT.csv](log/FastVim_tiny_val_ema_IN1k_supervised.csv) |
| [FastVim-S.ckpt](https://github.com/insitro/FastVim/releases/download/v0/FastVim_small.ckpt)    |       26M       |   81.1   | [FastVimS.yaml](config/FastVimS.yaml) | [FastVimS.csv](log/FastVim_small_val_ema_IN1k_supervised.csv) |
| [FastVim-B.ckpt](https://github.com/insitro/FastVim/releases/download/v0/FastVim_base.ckpt)    |       98M       |   82.6   | [FastVimB.yaml](config/FastVimB.yaml) | [FastVimB.csv](log/FastVim_base_val_ema_IN1k_supervised.csv) |
| [Vim-B w/ LN.ckpt](https://github.com/insitro/FastVim/releases/download/v0/Vim_base_withlayernorm.ckpt)    |       98M       |   82.6   | [VimB.yaml](config/VimB.yaml) | [VimB.csv](log/Vim_base_val_ema_IN1k_supervised.csv) |


**Notes:**
- For reproducibility, make sure overall batch size remains 1024 across GPUs/Nodes.
- trainer/global_step in log files refers to gradient steps with batch size 1024.
- Modify `imagenet_train_dir_path` and `imagenet_val_dir_path` flags in [datasets_supervised.py](datasets_supervised.py) code.
