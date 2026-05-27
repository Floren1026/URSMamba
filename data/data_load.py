import os
import glob
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

import random
import rasterio
import numpy as np
from natsort import natsorted


def load_data_rgb(config, dir, batchsize, flip=False, order=False):
    if flip:
        transforms = T.Compose([
                    # T.RandomCrop(config.DATA.IMG_SIZE),    
                    T.Resize([config.imageSize, config.imageSize]),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.ToTensor(),
                ])
    else:
        transforms = T.Compose([
                    # T.RandomCrop(config.DATA.IMG_SIZE),    
                    T.Resize([config.imageSize, config.imageSize]),
                    T.ToTensor(),
                ])
        

    dataset = ImageFolder(dir, transforms)
    assert dataset 

    if order:
        dataloader = DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=True,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=True
        ) 

    return dataloader


def load_data_multi(config, dir, batchsize):
    crop_size = config.imageSize
    dim = config.channel_secret

    dataloader = DataLoader(
        TiffImageDataset(dir, crop_size, dim),
        batch_size=batchsize,
        shuffle=True,
        pin_memory=False,
        num_workers=config.num_workers,
        drop_last=True
    )

    return dataloader


class TiffImageDataset(Dataset):
    def __init__(self, root_dir, crop_size=256, dim=8, crop_type='random'):

        self.root_dir = root_dir
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        self.dim = dim        
        self.crop_type = crop_type
        
        self.files = natsorted(sorted(glob.glob(root_dir + "/*." + 'tif')))


    def random_crop(self, image, crop_size):
        """
        image: torch.Tensor, shape (C, H, W)
        crop_size: (crop_h, crop_w)
        """
        C, H, W = image.shape
        crop_h, crop_w = crop_size
        assert H >= crop_h and W >= crop_w, "Crop size must be <= image size"

        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        cropped = image[:, top:top + crop_h, left:left + crop_w]
        return cropped
        

    def stretching(self, image):
        channels = image.shape[0]
        band_list = []
        for i in range(channels):
            band_data = image[i,:,:]
            band_min = np.percentile(band_data,2)
            band_max = np.percentile(band_data,98)

            band_range = band_max - band_min
            if band_range != 0:
                band_data = (band_data - band_min) / band_range
            else:
                band_data = np.zeros_like(band_data)

            # band_data = (band_data - band_min) / (band_max - band_min)
            # plt.imshow(band_data)
            # plt.show()
            band_list.append(band_data)
        image_data = np.stack(band_list, axis=0)
        image_data = np.clip(image_data, 0, 1)
        image = image_data.astype(np.float32)  # uint16 to float32
        image = torch.from_numpy(image)

        return image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 读取 TIFF 文件
        with rasterio.open(self.files[idx]) as src:
            image = src.read(list(range(1, self.dim + 1)))  # C, H, W

            image = self.random_crop(image, self.crop_size)
            image = self.stretching(image)

        return image