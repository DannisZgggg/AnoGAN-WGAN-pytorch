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
import jpeg4py as jpeg

from .albumentations import *
from .albumentations.pytorch import ToTensor




##
def provider(phase,category,batch_size=8,num_workers=4):
    dataset = MTVEC_Dataset(phase,category)
    df = dataset.get_csv()

    if phase == 'train':
        sampler = RandomSampler(df)
    else:
        sampler = SequentialSampler(df)
    dataloader = DataLoader(
        dataset,batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True)
    return dataloader



class MTVEC_Dataset(Dataset):
    def __init__(self,phase,category):
        super(MTVEC_Dataset, self).__init__()
        self.phase = phase
        self.category = category
        self.transforms = self.get_transforms(self.phase)
        self.df = self.generate_csv(self.phase,self.category)

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, int) else idx.item()
        image_path = self.df.iloc[idx]['image_path']
        label = int(self.df.iloc[idx]['label'])

        # image = jpeg.JPEG(image_path).decode()
        image = cv2.imread(image_path)
        image = self.transforms(image=image)["image"]  # [c,h,w]

        # TODO debug
        # cv2.imwrite('debug/'+str(idx)+'.jpg',image)
        # image = image[0].unsqueeze(0) #gray
        return image, label


    def __len__(self):
        return len(self.df)

    def get_transforms(self,phase):
        list_transforms = []
        list_transforms.extend([Resize(p=1,height=256,width=256)]) #512
        if phase == 'train':
            list_transforms.extend([
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    # RandomBrightnessContrast(p=0.2,brightness_limit=0.05,contrast_limit=0.05),
                    ShiftScaleRotate(p=0.2, shift_limit=(0.000,0.000), scale_limit=0,rotate_limit=10) # 3
                ])
        list_transforms.extend([ToTensor()]) #ToGray(p=1),

        list_trfms = Compose(list_transforms)
        return list_trfms


    def generate_csv(self, phase, category):
        image_path = []
        label = []

        if phase == 'train':
            train_root_path = 'mtvec_dataset/'+category+'/train/good'
            for img in os.listdir(train_root_path):
                image_path.append(os.path.join(train_root_path,img))
                label.append(0) #normal

        elif phase == 'test':
            test_root_path = 'mtvec_dataset/'+category+'/test'
            normal_class = 'good'
            abnormal_class = os.listdir(test_root_path)
            abnormal_class.remove(normal_class)

            for img in os.listdir(os.path.join(test_root_path, normal_class)):
                image_path.append(os.path.join(test_root_path, normal_class, img))
                label.append(0)  # normal

            for abnormal_name in abnormal_class:
                for img in os.listdir(os.path.join(test_root_path, abnormal_name)):
                    image_path.append(os.path.join(test_root_path, abnormal_name, img))
                    label.append(1)

        df = {'image_path': image_path,'label': label}
        df = pd.DataFrame(df)
        if not os.path.exists('df_dir'):
            os.mkdir('df_dir')
        df.to_csv('df_dir/'+category+'_'+phase+'.csv')
        # print(category+'_'+phase+'.csv generated...')
        self.df = df
        return df

    def get_csv(self):
        return self.df
