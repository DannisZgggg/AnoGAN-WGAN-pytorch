"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from PIL import Image
import cv2
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler
import numpy as np
import pandas as pd
# import jpeg4py as jpeg

from .albumentations import *
from .albumentations.pytorch import ToTensor

from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.utils import check_integrity



##
def provider(phase, category, batch_size=8, num_workers=4):
    dataset = CIFAR10_abnrmal(phase, category)
    if phase == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True)
    return dataloader


class CIFAR10_abnrmal(Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, phase='train', target_class='zero'):
        self.root = 'data/cifar'
        self.transform = self.get_transforms(phase)
        self.phase = phase

        str_2_int = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                     'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
        target_class = str_2_int[target_class]

        self.ori_train_data = []
        self.ori_train_target = []

        # now load the picked numpy arrays
        for file_name, checksum in self.train_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.ori_train_data.append(entry['data'])

                if 'labels' in entry:
                    self.ori_train_target.extend(entry['labels'])
                else:
                    self.ori_train_target.extend(entry['fine_labels'])

        self.ori_train_data = np.vstack(self.ori_train_data).reshape(-1, 3, 32, 32)
        self.ori_train_data = self.ori_train_data.transpose((0, 2, 3, 1))  # convert to HWC
        self.ori_train_target = np.array(self.ori_train_target)

        self.ori_test_data = []
        self.ori_test_target = []

        # now load the picked numpy arrays
        for file_name, checksum in self.test_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.ori_test_data.append(entry['data'])

                if 'labels' in entry:
                    self.ori_test_target.extend(entry['labels'])
                else:
                    self.ori_test_target.extend(entry['fine_labels'])

        self.ori_test_data = np.vstack(self.ori_test_data).reshape(-1, 3, 32, 32)
        self.ori_test_data = self.ori_test_data.transpose((0, 2, 3, 1))  # convert to HWC
        self.ori_test_target = np.array(self.ori_test_target)

        self._load_meta()

        self.train_data, self.train_target, self.test_data, self.test_target =\
            self.get_cifar_unimodal_dataset(self.ori_train_data, self.ori_train_target,
                                       self.ori_test_data, self.ori_test_target,target_class)

        if self.phase == 'train':
            self.data, self.target = self.train_data, self.train_target
        else:
            self.data, self.target = self.test_data, self.test_target
        assert len(self.data) == len(self.target)


    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        # 'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
        # 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9


    def get_transforms(self, phase):
        list_transforms = []
        # list_transforms.extend([Resize(p=1, height=32, width=32)])  # 512
        if phase == 'train':
            list_transforms.extend([
                HorizontalFlip(p=0.5)
                # ShiftScaleRotate(p=0.5, shift_limit=(0.01, 0.01), scale_limit=0.1, rotate_limit=10)
            ])
        list_transforms.extend([ToTensor()])

        list_trfms = Compose(list_transforms)
        return list_trfms


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = self.transform(image=img)["image"]

        return img, target

    def __len__(self):
        return len(self.data)


    ##
    @staticmethod
    def get_cifar_unimodal_dataset(trn_img, trn_lbl, tst_img, tst_lbl, nrm_cls_idx=1, proportion=0.5):
        """ Create mnist 2 anomaly dataset.
        Arguments:
            trn_img {np.array} -- Training images
            trn_lbl {np.array} -- Training labels
            tst_img {np.array} -- Test     images
            tst_lbl {np.array} -- Test     labels
        Keyword Arguments:
            nrm_cls_idx {int} -- Anomalous class index (default: {0})
        Returns:
            [tensor] -- New training-test images and labels.
        """

        # --
        # Find normal abnormal indexes.
        # TODO: PyTorch v0.4 has torch.where function
        nrm_trn_idx = np.where(trn_lbl == nrm_cls_idx)[0]  # 5000
        abn_trn_idx = np.where(trn_lbl != nrm_cls_idx)[0]  # 45000
        nrm_tst_idx = np.where(tst_lbl == nrm_cls_idx)[0]  # 1000
        abn_tst_idx = np.where(tst_lbl != nrm_cls_idx)[0]  # 9000

        # Get n percent of the abnormal samples.
        idx = np.arange(len(abn_tst_idx))
        np.random.shuffle(abn_tst_idx)

        abn_tst_idx = abn_tst_idx[idx]
        abn_tst_idx = abn_tst_idx[:int(len(idx) * proportion)]  # 4500

        # --
        # Find normal and abnormal images
        nrm_trn_img = trn_img[nrm_trn_idx]  # Normal training images
        abn_trn_img = trn_img[abn_trn_idx]  # Abnormal training images.
        nrm_tst_img = tst_img[nrm_tst_idx]  # Normal test images
        abn_tst_img = tst_img[abn_tst_idx]  # Abnormal test images.

        # --
        # Find normal and abnormal labels.
        nrm_trn_lbl = trn_lbl[nrm_trn_idx]  # Normal training labels
        abn_trn_lbl = trn_lbl[abn_trn_idx]  # Abnormal training labels.
        nrm_tst_lbl = tst_lbl[nrm_tst_idx]  # Normal test labels
        abn_tst_lbl = tst_lbl[abn_tst_idx]  # Abnormal test labels.

        # --
        # Assign labels to normal (0) and abnormals (1)
        nrm_trn_lbl[:] = 0
        nrm_tst_lbl[:] = 0
        abn_trn_lbl[:] = 1
        abn_tst_lbl[:] = 1

        # Create new anomaly dataset based on the following data structure:
        new_trn_img = nrm_trn_img.copy()
        new_trn_lbl = nrm_trn_lbl.copy()  # 5000
        new_tst_img = np.concatenate((nrm_tst_img, abn_tst_img), axis=0)
        new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_tst_lbl), axis=0)
        return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl


if __name__ == '__main__':
    data = CIFAR10_abnrmal(phase='test')
    # print(len(data))
    print(data[1000][0].shape,data[1000][1])

