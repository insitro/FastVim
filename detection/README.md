## Dataset

To download and prepare the MSCOCO 2017 dataset, execute the following commands:

```
wget http://images.cocodataset.org/zips/train2017.zip -P coco
wget http://images.cocodataset.org/zips/val2017.zip -P coco
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P coco
cd coco
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip
rm train2017.zip
rm val2017.zip
rm annotations_trainval2017.zip
```

Then in config/FastVim/lsj-100e_coco-instance.py, change data_root path to dataset

## Training

```
chmod +x ./detection/tools/dist_train.sh

FastVimT: ./detection/tools/dist_train.sh "detection/configs/FastVim/vitdet_cascade_mask-rcnn_FastVim_tiny_noclstok_rotate_layernorm_lsj-300e.py" 8 "path_to_imagenet_supervised_ckpt" 

FastVimS: ./detection/tools/dist_train.sh "detection/configs/FastVim/vitdet_cascade_mask-rcnn_FastVim_small_noclstok_rotate_layernorm_lsj-300e.py" 8 "path_to_imagenet_supervised_ckpt" 

FastVimB: ./detection/tools/dist_train.sh "detection/configs/FastVim/vitdet_cascade_mask-rcnn_FastVim_base_noclstok_rotate_layernorm_lsj-300e.py" 8 "path_to_imagenet_supervised_ckpt" 

VimS: ./detection/tools/dist_train.sh "detection/configs/FastVim/vitdet_cascade_mask-rcnn_Vim_small_lsj-300e.py" 8 "path_to_imagenet_supervised_ckpt" 
```

## Model weights and configurations

| Model | AP<sup>box</sup> |
|:------------------------------------------------------------------:|:----------:|
| [FastVim-T.ckpt](https://github.com/insitro/FastVim/releases/download/v0/FastVim_tiny_iter_184375.pth)    |   45.1   |
| [FastVim-S.ckpt](https://github.com/insitro/FastVim/releases/download/v0/FastVim_small_iter_184375.pth)    |   48.4   |
| [FastVim-B.ckpt](https://github.com/insitro/FastVim/releases/download/v0/FastVim_base_iter_184375.pth)    |   50.0   |
| [Vim-S.ckpt](https://github.com/insitro/FastVim/releases/download/v0/Vim_small_transferfromgithub_iter_184375.pth)    |    47.1   |



