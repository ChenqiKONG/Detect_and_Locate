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
        msk_path = img_path.split('faces_224')[0] + 'masks_224' + img_path.split('faces_224')[1].split('face')[0] + img_path.split('faces_224')[1].split('face')[1]
        nis_path =  img_path.split('FF++_faces_masks_labels')[0] + 'FF++_faces_masks_labels_noise' + img_path.split('FF++_faces_masks_labels')[1].split('faces_224')[0] + img_path.split('FF++_faces_masks_labels')[1].split('faces_224')[1].split('c23')[0] + 'c23/Sigma5' + img_path.split('FF++_faces_masks_labels')[1].split('faces_224')[1].split('c23')[1]
        img = io.imread(img_path)
        msk = io.imread(msk_path)
        nis = io.imread(nis_path)
        label = self.data_path_details.iloc[idx, 1]

        sample = {'face': img,
                  'mask':msk,
                  'noise':nis,
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class my_transforms(object):
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
    """
    def __init__(self, output_size=299, msk_size1= 147,  msk_size2= 37,  msk_size3= 19, RandomHorizontalFlip=False, mean=0.5, std=0.5):

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
            self.mask_size1 = (msk_size1, msk_size1)
            self.mask_size2 = (msk_size2, msk_size2)
            self.mask_size3 = (msk_size3, msk_size3)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.RandomHorizontalFlip = RandomHorizontalFlip
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_ori = sample['face']
        msk_ori = sample['mask']
        nis_ori_ = sample['noise']
        label = sample['label']
        nis_ori = nis_ori_[:,:,np.newaxis]

        if len(msk_ori.shape) != 3:
            msk_ori = msk_ori[:,:,np.newaxis]
        ### resize
        img = transform.resize(img_ori, self.output_size)
        msk1 = transform.resize(msk_ori, self.mask_size1)
        msk2 = transform.resize(msk_ori, self.mask_size2)
        msk3 = transform.resize(msk_ori, self.mask_size3)
        nis1 = transform.resize(nis_ori, self.mask_size1)
        nis2 = transform.resize(nis_ori, self.mask_size2)
        nis3 = transform.resize(nis_ori, self.mask_size3)

        msk1 = np.mean(msk1.transpose(2, 0, 1),axis=0)
        msk2 = np.mean(msk2.transpose(2, 0, 1),axis=0)
        msk3 = np.mean(msk3.transpose(2, 0, 1),axis=0)

        nis_1 = nis1.transpose(2, 0, 1)
        nis_2 = nis2.transpose(2, 0, 1)
        nis_3 = nis3.transpose(2, 0, 1)

        ### random horizontal flip
        if self.RandomHorizontalFlip and random.random() < 0.5:
            img = img[:, ::-1, :]

        ### (h, w, c) -> (c, h, w)
        img = img.transpose(2, 0, 1)
        msk_1 = np.concatenate((msk1[np.newaxis, :, :], (1.0 - msk1)[np.newaxis, :, :]), axis=0)
        msk_2 = np.concatenate((msk2[np.newaxis, :, :], (1.0 - msk2)[np.newaxis, :, :]), axis=0)
        msk_3 = np.concatenate((msk3[np.newaxis, :, :], (1.0 - msk3)[np.newaxis, :, :]), axis=0)

        ### to pytorch tensor
        img = torch.from_numpy(img.copy()).float()
        label_msk1 = torch.tensor(msk_1).float()
        label_msk2 = torch.tensor(msk_2).float()
        label_msk3 = torch.tensor(msk_3).float()
        label_nis1 = torch.tensor(nis_1).float()
        label_nis2 = torch.tensor(nis_2).float()
        label_nis3 = torch.tensor(nis_3).float()
        label = torch.tensor(label).float()

        ### normalize
        img = (img - self.mean) / self.std

        ### output
        sample = {'faces': img,
                  'masks1':label_msk1,
                  'masks2':label_msk2,
                  'masks3':label_msk3,
                  'noises1':label_nis1,
                  'noises2':label_nis2,
                  'noises3':label_nis3,
                  'labels': label}
        return sample