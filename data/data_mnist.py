"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
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

from torchvision.datasets import MNIST


##
def provider(phase, category, batch_size=8, num_workers=4):
    dataset = MNIST_abnormal(phase, category)
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


class MNIST_abnormal(Dataset):
    def __init__(self, phase, target_class='zero', multimodal=False):
        super(MNIST_abnormal, self).__init__()
        # treating one class being an anomaly, while the rest of the
        # classes are considered as the normal class
        self.phase = phase
        self.transforms = self.get_transforms(self.phase)
        str_2_int = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                     'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
        # 6903 7877 6990 7141 6824 6313 6876 7293 6825 6958
        target_class = str_2_int[target_class]

        root = 'data/mnist'
        ori_mnist_dataset = MNIST(root)
        if not ori_mnist_dataset._check_exists():
            ori_mnist_dataset.download()
        del ori_mnist_dataset

        ori_train_data, ori_train_targets = \
            torch.load(os.path.join(root, 'MNIST', 'processed', 'training.pt'))
        ori_test_data, ori_test_targets = \
            torch.load(os.path.join(root, 'MNIST', 'processed', 'test.pt'))  # torch.Tensor

        if multimodal:
            self.train_data, self.train_target,self.test_data, self.test_target = \
                self.get_mnist_multimodal_dataset(ori_train_data, ori_train_targets,
                                             ori_test_data, ori_test_targets,target_class)

        else:
            self.train_data, self.train_target, self.test_data, self.test_target = \
                self.get_mnist_unimodal_dataset(ori_train_data, ori_train_targets,
                                             ori_test_data, ori_test_targets,target_class)


        if self.phase == 'train':
            self.data, self.target = self.train_data, self.train_target
        else:
            self.data, self.target = self.test_data, self.test_target
        assert len(self.data) == len(self.target)

    def get_transforms(self, phase):
        list_transforms = []
        list_transforms.extend([Resize(p=1, height=32, width=32)])  # 512  TODO
        if phase == 'train':
            list_transforms.extend([
                ShiftScaleRotate(p=0.5, shift_limit=(0.01, 0.01), scale_limit=0.1, rotate_limit=10)
            ])
        list_transforms.extend([ToTensor()])

        list_trfms = Compose(list_transforms)
        return list_trfms

    def __getitem__(self, index):
        image, target = self.data[index], int(self.target[index])
        image = np.expand_dims(image.numpy(), axis=-1)
        image = self.transforms(image=image)["image"]  # [c,h,w]
        # cv2.imwrite('./debug/'+str(index)+'.jpg',image)
        return image, target

    def __len__(self):
        return len(self.data)


    ##
    @staticmethod
    def get_mnist_multimodal_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0):
        """[summary]
        Arguments:
            trn_img {np.array} -- Training images
            trn_lbl {np.array} -- Training labels
            tst_img {np.array} -- Test     images
            tst_lbl {np.array} -- Test     labels
        Keyword Arguments:
            abn_cls_idx {int} -- Anomalous class index (default: {0})
        Returns:
            [np.array] -- New training-test images and labels.
        """
        # --
        # Find normal abnormal indexes.
        nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
        abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
        nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
        abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

        # --
        # Find normal and abnormal images
        nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
        abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
        nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
        abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

        # --
        # Find normal and abnormal labels.
        nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
        abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
        nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
        abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

        # --
        # Assign labels to normal (0) and abnormals (1)
        nrm_trn_lbl[:] = 0
        nrm_tst_lbl[:] = 0
        abn_trn_lbl[:] = 1
        abn_tst_lbl[:] = 1

        # Concatenate the original train and test sets.
        nrm_img = torch.cat((nrm_trn_img, nrm_tst_img), dim=0)
        nrm_lbl = torch.cat((nrm_trn_lbl, nrm_tst_lbl), dim=0)
        abn_img = torch.cat((abn_trn_img, abn_tst_img), dim=0)
        abn_lbl = torch.cat((abn_trn_lbl, abn_tst_lbl), dim=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

        # Create new anomaly dataset based on the following data structure:
        new_trn_img = nrm_trn_img.clone()
        new_trn_lbl = nrm_trn_lbl.clone()
        new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
        new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

        return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

    ##
    @staticmethod
    def get_mnist_unimodal_dataset(trn_img, trn_lbl, tst_img, tst_lbl, nrm_cls_idx=0, proportion=0.5):
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
        nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == nrm_cls_idx)[0])
        abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != nrm_cls_idx)[0])
        nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == nrm_cls_idx)[0])
        abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != nrm_cls_idx)[0])

        # Get n percent of the abnormal samples.
        abn_tst_idx = abn_tst_idx[torch.randperm(len(abn_tst_idx))]
        abn_tst_idx = abn_tst_idx[:int(len(abn_tst_idx) * proportion)]


        # --
        # Find normal and abnormal images
        nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
        abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
        nrm_tst_img = tst_img[nrm_tst_idx]    # Normal test images
        abn_tst_img = tst_img[abn_tst_idx]    # Abnormal test images.

        # --
        # Find normal and abnormal labels.
        nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
        abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
        nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal test labels
        abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal test labels.

        # --
        # Assign labels to normal (0) and abnormals (1)
        nrm_trn_lbl[:] = 0
        nrm_tst_lbl[:] = 0
        abn_trn_lbl[:] = 1
        abn_tst_lbl[:] = 1

        # Create new anomaly dataset based on the following data structure:
        new_trn_img = nrm_trn_img.clone()
        new_trn_lbl = nrm_trn_lbl.clone()
        new_tst_img = torch.cat((nrm_tst_img, abn_tst_img), dim=0)
        new_tst_lbl = torch.cat((nrm_tst_lbl, abn_tst_lbl), dim=0)

        # added
        idx = torch.randint(new_tst_lbl.shape[0],(100,))
        new_tst_img = new_tst_img[idx]
        new_tst_lbl = new_tst_lbl[idx]

        return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl
