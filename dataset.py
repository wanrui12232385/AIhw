import torch
import numpy as np
import random
from torch.utils.data import Dataset
import soundfile as sf

root = "/Users/wangrui/Downloads/LibriSpeech-SI/train"

class MyDataset(Dataset):

    def __init__(self, voice, labels, transform=None, target_transform=None):

        self.voice = voice
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.voice)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.voice[idx]
        target = self.labels[idx]
        voice = sf.read(path)[0]

        if self.transform:
            voice = self.transform(voice)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return voice, target
