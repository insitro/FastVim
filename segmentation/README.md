## Dataset

To download and prepare the ADE20K dataset, execute the following commands:

```
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip -q ADEChallengeData2016.zip
rm ADEChallengeData2016.zip
```

Then in config/_base_/datasets/ade20k.py, change data_root path to dataset

## Training
 ```
FastVimT: ./segmentation/tools/dist_train.sh "segmentation/configs/FastVim/uppernet_FastVim_tiny_noclstok_rotate_layernorm_8xb2-160k_ade20k-512x512.py" 8 "path_to_imagenet_supervised_ckpt" 

FastVimS: ./segmentation/tools/dist_train.sh "segmentation/configs/FastVim/uppernet_FastVim_small_noclstok_rotate_layernorm_8xb2-160k_ade20k-512x512.py" 8 "path_to_imagenet_supervised_ckpt" 

FastVimB: ./segmentation/tools/dist_train.sh "segmentation/configs/FastVim/uppernet_FastVim_base_noclstok_rotate_layernorm_8xb2-160k_ade20k-512x512.py" 8 "path_to_imagenet_supervised_ckpt" 

```

## Model weights and configurations

| Model | mIOU |
|:------------------------------------------------------------------:|:----------:|
| [FastVim-T.ckpt](https://github.com/insitro/FastVim/releases/download/v0/FastVim_tiny_iter_160000.pth)    |   41.8  |
| [FastVim-S.ckpt](https://github.com/insitro/FastVim/releases/download/v0/FastVim_small_iter_160000.pth)    |   44.9   |
| [FastVim-B.ckpt](https://github.com/insitro/FastVim/releases/download/v0/FastVim_base_iter_160000.pth)    |   47.8   |
