import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset

class face_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_path_details = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data_path_details)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_path_details.iloc[idx, 0]
        input_img = io.imread(img_path)

        noise_path = self.data_path_details.iloc[idx, 1]
        noise = io.imread(noise_path)

        mask_path = self.data_path_details.iloc[idx, 2]
        mask = io.imread(mask_path)

        label = self.data_path_details.iloc[idx, 3]

        sample = {'img': input_img,
                  'noise': noise,
                  'mask': mask,
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class my_transforms(object):
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
    """

    def __init__(self, size1=299, size2=147, size3=37, size4=19, RandomHorizontalFlip=False, mean=0.5, std=0.5):

        if isinstance(size1, int):
            self.size1 = (size1, size1)
            self.size2 = (size2, size2)
            self.size3 = (size3, size3)
            self.size4 = (size4, size4)

        self.RandomHorizontalFlip = RandomHorizontalFlip

        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_ori = sample['img']
        label_noise = sample['noise']
        label_mask = sample['mask']
        label_fake = sample['label']
        label_noise = label_noise[:, :, np.newaxis]
        if len(label_mask.shape) != 3:
            label_mask = label_mask[:, :, np.newaxis]

        face = transform.resize(img_ori, self.size1)
        noise_seg1 = transform.resize(label_noise, self.size2)
        noise_seg2 = transform.resize(label_noise, self.size3)
        noise_seg3 = transform.resize(label_noise, self.size4)
        mask_seg1 = transform.resize(label_mask, self.size2)
        mask_seg2 = transform.resize(label_mask, self.size3)
        mask_seg3 = transform.resize(label_mask, self.size4)

        ### (h, w, c) -> (c, h, w)
        face = face.transpose(2, 0, 1)
        noise_seg1 = noise_seg1.transpose(2, 0, 1)
        noise_seg2 = noise_seg2.transpose(2, 0, 1)
        noise_seg3 = noise_seg3.transpose(2, 0, 1)

        mask_seg1 = np.mean(mask_seg1.transpose(2, 0, 1), axis=0)
        mask_seg2 = np.mean(mask_seg2.transpose(2, 0, 1), axis=0)
        mask_seg3 = np.mean(mask_seg3.transpose(2, 0, 1), axis=0)

        mask_seg_1 = np.concatenate((mask_seg1[np.newaxis, :, :], (1.0 - mask_seg1)[np.newaxis, :, :]), axis=0)
        mask_seg_2 = np.concatenate((mask_seg2[np.newaxis, :, :], (1.0 - mask_seg2)[np.newaxis, :, :]), axis=0)
        mask_seg_3 = np.concatenate((mask_seg3[np.newaxis, :, :], (1.0 - mask_seg3)[np.newaxis, :, :]), axis=0)

        ### to pytorch tensor
        face = torch.from_numpy(face.copy()).float()
        label_noise_seg1 = torch.tensor(noise_seg1).float()
        label_noise_seg2 = torch.tensor(noise_seg2).float()
        label_noise_seg3 = torch.tensor(noise_seg3).float()
        label_mask_seg1 = torch.tensor(mask_seg_1).float()
        label_mask_seg2 = torch.tensor(mask_seg_2).float()
        label_mask_seg3 = torch.tensor(mask_seg_3).float()
        label_fake = torch.tensor(label_fake).float()

        ### normalize
        face = (face - self.mean) / self.std

        ### output
        sample_tran = {'faces': face,
                       'label_noise_seg1': label_noise_seg1,
                       'label_noise_seg2': label_noise_seg2,
                       'label_noise_seg3': label_noise_seg3,
                       'label_mask_seg1': label_mask_seg1,
                       'label_mask_seg2': label_mask_seg2,
                       'label_mask_seg3': label_mask_seg3,
                       'labels': label_fake}

        return sample_tran