import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class face_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_path_details = pd.read_csv(csv_file,header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data_path_details)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_path_details.iloc[idx, 0]
        mask_path = self.data_path_details.iloc[idx, 2]

        img = io.imread(img_path)
        mask = io.imread(mask_path)

        label = self.data_path_details.iloc[idx, 3]

        sample = {'face': img,
                  'mask' : mask,
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class my_transforms(object):
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
    """
    def __init__(self, output_size=256, mask_size=256, RandomHorizontalFlip=False, mean=0.5, std=0.5):

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
            self.mask_size = (mask_size, mask_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.RandomHorizontalFlip = RandomHorizontalFlip
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_ori = sample['face']
        mask_ori = sample['mask']
        label = sample['label']

        ### resize
        img = transform.resize(img_ori, self.output_size)
        if len(mask_ori.shape) != 3:
            mask_ori = mask_ori[:,:,np.newaxis]

        mask = transform.resize(mask_ori, self.mask_size)

        ### random horizontal flip
        if self.RandomHorizontalFlip and random.random() < 0.5:
            img = img[:, ::-1, :]

        ### (h, w, c) -> (c, h, w)
        img = img.transpose(2, 0, 1)

        mask = np.mean(mask.transpose(2, 0, 1),axis=0)
        mask = np.concatenate((mask[np.newaxis, :, :], (1.0 - mask)[np.newaxis, :, :]), axis=0)

        ### to pytorch tensor
        img = torch.from_numpy(img.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()
        label = torch.tensor(label).float()

        ### normalize
        img = (img - self.mean) / self.std

        ### output
        sample = {'faces': img,
                  'masks': mask,
                  'labels': label}
        return sample