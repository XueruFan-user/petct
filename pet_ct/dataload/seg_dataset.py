import os
import pandas as pd
from PIL import Image
import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt




def ImageTrans512(prob):
    train_transformI = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation((-15, 15),
                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomResizedCrop(512,scale=(0.8,1),ratio=(0.8,1),
                                                 interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomHorizontalFlip(p=prob*0.2),
        torchvision.transforms.RandomVerticalFlip(p=prob * 0.1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transformI


def ROITrans512(prob):
    train_transformI = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation((-15, 15),
                                              interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        torchvision.transforms.RandomResizedCrop(512, scale=(0.8, 1), ratio=(0.8, 1),
                                                 interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        torchvision.transforms.RandomHorizontalFlip(p=prob * 0.2),
        torchvision.transforms.RandomVerticalFlip(p=prob * 0.1),
        torchvision.transforms.ToTensor(),
    ])
    return train_transformI

def vali_imagetrans512():
    train_transformI = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512),
                                      interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transformI

def vali_roitrans512():
    train_transformI = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512),
                                      interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        torchvision.transforms.ToTensor(),
    ])
    return train_transformI

def ImageTrans128(prob):
    train_transformI = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation((-15, 15),
                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomResizedCrop(128,scale=(0.8,1),ratio=(0.8,1),
                                                 interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomHorizontalFlip(p=prob*0.2),
        torchvision.transforms.RandomVerticalFlip(p=prob * 0.1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transformI


def vali_imagetrans128():
    train_transformI = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128),
                                      interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transformI


class MyDataSet(Dataset):
    def __init__(self, img_path, data_list, transform = True):
        self.img_path = img_path
        self.data_list = data_list
        self.transform = transform
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, str(self.data_list[index]))
        data = np.load(img_path)
        img = data['image']
        roi = data['mask']

        pro = 0.5

        if self.transform:
            transformI512 = ImageTrans512(pro)
            transformI128 = ImageTrans128(pro)
            transformR512 = ROITrans512(pro)
        else:
            transformI512 = vali_imagetrans512()
            transformI128 = vali_imagetrans128()
            transformR512 = vali_roitrans512()

        seed = np.random.randint(0, 2 ** 16)

        img = Image.fromarray(img)

        random.seed(seed)
        torch.manual_seed(seed)

        img512 = transformI512(img)

        random.seed(seed)
        torch.manual_seed(seed)

        img128 = transformI128(img)

        random.seed(seed)
        torch.manual_seed(seed)

        roi = transformR512(roi)

        return img128, img512, roi

