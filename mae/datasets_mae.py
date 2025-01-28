import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy
from torchvision import datasets, transforms

IMAGENET_MEAN_PER_CHANNEL = (0.485, 0.456, 0.406)
IMAGENET_STD_PER_CHANNEL = (0.229, 0.224, 0.225)

imagenet_train_dir_path = "path to train dir"


class create_imagenet_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_size,
        hflip,
        interpolation,
        batch_size: int = 64,
        num_workers: int = 12,
    ):
        super().__init__()

        self.transform_train = RGBAugmentation(
            img_size=img_size,
            hflip=hflip,
            interpolation=interpolation,
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

        # for CIFAR dataset
        # self.dataset_train = datasets.CIFAR100('/path/to/save', train=True, transform=self.transform_train, download=True)

        # for ImageNet-1k dataset
        self.dataset_train = datasets.ImageFolder(
            imagenet_train_dir_path, transform=self.transform_train
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


def load_DataModule(
    batch_size: int = 128,
    num_workers: int = 1,
    img_size=224,
    hflip=0.5,
    interpolation="bicubic",
) -> pl.LightningDataModule:
    """Load the Imagenet dataset."""

    return create_imagenet_DataModule(
        img_size, hflip, interpolation, batch_size, num_workers
    )


class RGBAugmentation:
    """Transformations for standard RBG images"""

    def __init__(
        self,
        img_size: int = 224,
        hflip: float = 0.5,
        scale=None,
        ratio=None,
        interpolation="bicubic",
        use_prefetcher=False,
        mean=IMAGENET_MEAN_PER_CHANNEL,
        std=IMAGENET_STD_PER_CHANNEL,
    ):
        scale = tuple(scale or (0.2, 1.0))  # default imagenet scale range
        ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, ratio=ratio, interpolation=interpolation
            )
        ]
        if hflip > 0.0:
            primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]

        final_tfl = []
        if use_prefetcher:
            # prefetcher and collate will handle tensor conversion and norm
            final_tfl += [ToNumpy()]
        else:
            final_tfl += [
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]

        self.primary_transform = transforms.Compose(primary_tfl)
        self.final_transform = transforms.Compose(final_tfl)

    def __call__(self, img) -> torch.Tensor:
        img = Image.fromarray(np.asarray(img))
        img = self.primary_transform(img)
        img = self.final_transform(img)

        return img
