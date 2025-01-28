import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from timm.data.auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from timm.data.random_erasing import RandomErasing
from timm.data.transforms import (
    RandomResizedCropAndInterpolation,
    ToNumpy,
    str_to_pil_interp,
)
from torchvision import datasets, transforms

IMAGENET_MEAN_PER_CHANNEL = (0.485, 0.456, 0.406)
IMAGENET_STD_PER_CHANNEL = (0.229, 0.224, 0.225)


imagenet_train_dir_path = "path to train dir"
imagenet_val_dir_path = "path to val dir"


class create_imagenet_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_size,
        hflip,
        vflip,
        eval_crop_ratio,
        color_jitter,
        auto_augment,
        interpolation,
        re_prob,
        re_mode,
        re_count,
        batch_size: int = 64,
        num_workers: int = 12,
    ):
        super().__init__()

        self.transform_train = RGBAugmentation(
            is_train=True,
            img_size=img_size,
            hflip=hflip,
            vflip=vflip,
            interpolation=interpolation,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
        )

        self.transform_val = RGBAugmentation(
            is_train=False, img_size=img_size, eval_crop_ratio=eval_crop_ratio
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

        # for CIFAR dataset
        # self.dataset_train = datasets.CIFAR100('/path/to/save', train=True, transform=self.transform_train, download=True)
        # self.dataset_val = datasets.CIFAR100('/path/to/save', train=False, transform=self.transform_val)
        # self.dataset_test = datasets.CIFAR100('/path/to/save', train=False, transform=self.transform_val)

        # for ImageNet-1k dataset
        self.dataset_train = datasets.ImageFolder(
            imagenet_train_dir_path, transform=self.transform_train
        )
        self.dataset_val = datasets.ImageFolder(
            imagenet_val_dir_path, transform=self.transform_val
        )
        self.dataset_test = datasets.ImageFolder(
            imagenet_val_dir_path, transform=self.transform_val
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )


def load_DataModule(
    batch_size: int = 128,
    num_workers: int = 1,
    img_size=224,
    hflip=0.5,
    vflip=0.0,
    eval_crop_ratio=0.875,
    color_jitter=0.3,
    auto_augment=None,
    interpolation="bicubic",
    re_prob=0.0,
    re_mode="pixel",
    re_count=1,
) -> pl.LightningDataModule:
    """Load the Imagenet dataset."""

    return create_imagenet_DataModule(
        img_size,
        hflip,
        vflip,
        eval_crop_ratio,
        color_jitter,
        auto_augment,
        interpolation,
        re_prob,
        re_mode,
        re_count,
        batch_size,
        num_workers,
    )


class RGBAugmentation:
    """Transformations for standard RBG images"""

    def __init__(
        self,
        is_train: bool = False,
        img_size: int = 224,
        hflip: float = 0.5,
        vflip: float = 0.0,
        eval_crop_ratio=0.875,
        scale=None,
        ratio=None,
        color_jitter=0.0,
        auto_augment=None,
        interpolation="bicubic",
        use_prefetcher=False,
        mean=IMAGENET_MEAN_PER_CHANNEL,
        std=IMAGENET_STD_PER_CHANNEL,
        re_prob=0.0,
        re_mode="const",
        re_count=1,
        re_num_splits=0,
    ):
        self.is_train = is_train
        if is_train:
            scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
            ratio = tuple(
                ratio or (3.0 / 4.0, 4.0 / 3.0)
            )  # default imagenet ratio range
            primary_tfl = [
                RandomResizedCropAndInterpolation(
                    img_size, scale=scale, ratio=ratio, interpolation=interpolation
                )
            ]
            if hflip > 0.0:
                primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
            if vflip > 0.0:
                primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

            secondary_tfl = []
            if auto_augment:
                print("Auto augment on")
                assert isinstance(auto_augment, str)
                if isinstance(img_size, (tuple, list)):
                    img_size_min = min(img_size)
                else:
                    img_size_min = img_size
                aa_params = dict(
                    translate_const=int(img_size_min * 0.45),
                    img_mean=tuple([min(255, round(255 * x)) for x in mean]),
                )
                if interpolation and interpolation != "random":
                    aa_params["interpolation"] = str_to_pil_interp(interpolation)
                if auto_augment.startswith("rand"):
                    secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
                elif auto_augment.startswith("augmix"):
                    aa_params["translate_pct"] = 0.3
                    secondary_tfl += [
                        augment_and_mix_transform(auto_augment, aa_params)
                    ]
                else:
                    secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]

            # MAE removes color jitter in fine tuning and just use random aug
            # elif color_jitter is not None:
            #     # color jitter is enabled when not using AA
            #     if isinstance(color_jitter, (list, tuple)):
            #         # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            #         # or 4 if also augmenting hue
            #         assert len(color_jitter) in (3, 4)
            #     else:
            #         # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            #         color_jitter = (float(color_jitter),) * 3
            #     secondary_tfl += [transforms.ColorJitter(*color_jitter)]

            final_tfl = []
            if use_prefetcher:
                # prefetcher and collate will handle tensor conversion and norm
                final_tfl += [ToNumpy()]
            else:
                final_tfl += [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(mean), std=torch.tensor(std)
                    ),
                ]
                if re_prob > 0.0:
                    final_tfl.append(
                        RandomErasing(
                            re_prob,
                            mode=re_mode,
                            max_count=re_count,
                            num_splits=re_num_splits,
                            device="cpu",
                        )
                    )

            self.primary_transform = transforms.Compose(primary_tfl)
            self.secondary_transform = transforms.Compose(secondary_tfl)
            self.final_transform = transforms.Compose(final_tfl)

        else:
            t = []

            if img_size > 224:
                eval_crop_ratio = 1.0  # this setting taken from MAE paper

            size = int(img_size / eval_crop_ratio)
            t.append(
                transforms.Resize(
                    size, interpolation=3
                ),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(img_size))

            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            self.transform = transforms.Compose(t)

    def __call__(self, img) -> torch.Tensor:
        img = Image.fromarray(np.asarray(img))
        if self.is_train:
            img = self.primary_transform(img)
            try:
                img = self.secondary_transform(img)
            except Exception as e:
                print("-----------secondary_transform failed-----------", e)

            img = self.final_transform(img)
        else:
            img = self.transform(img)
        return img
