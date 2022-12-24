import torch
import numpy as np
import random
import os
from torch.utils.data import Dataset
import soundfile as sf
import random

root = "/Users/wangrui/Downloads/LibriSpeech-SI/train"
mask_root = "/Users/wangrui/Downloads/LibriSpeech-SI/noise"

def getmask(mask_root):
    mask = []
    for video in os.listdir(mask_root):
        mask.append(sf.read(os.path.join(mask_root,video)))
    return getmask()

class MyDataset(Dataset):

    def __init__(self, voice, labels, mask,transform=None, target_transform=None):

        self.voice = voice
        self.labels = labels
        self.transform = transform
        self.mask = mask
        self.target_transform = target_transform

    def __len__(self):
        return len(self.voice)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.voice[idx]
        target = self.labels[idx]
        voice = sf.read(path)[0]
        length = min(len(voice),160000)
        voice = voice[0:length]
        a = random.randint(0,len(self.mask)-1)
        m = self.mask[a]
        m = m[0:length]
        voice = voice+random.uniform(0.13,0.20)*m
        if self.transform:
            voice = self.transform(voice)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return voice, target
