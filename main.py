import torch
from dataset import MyDataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch import nn
import os
import numpy as np


VOICE_EXTENSIONS = ('.flac', '.wav')

def is_voice_file(filename):
    return  filename.lower().endswith(VOICE_EXTENSIONS)

def train(train_dataset,test_dataset, batch_size, workers, model):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=workers)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=workers)

    loss_function = nn.CrossEntropyLoss()

    lr = 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    epoch = 10000

    for k in range(epoch):
        for data in train_dataloader:
            voice, label = data
            output = model(voice)
            loss = loss_function(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if k % 1000 == 0:
            print('epoch:{}, loss:{}'.format(k,loss.data.item()))
            with torch.no_grad():
                length = len(test_dataloader)
                sum = 0
                for data in test_dataloader:
                    voice, label = data
                    output = model(voice)
                    max = torch.max(output)
                    output = torch.where(output == max, 1, 0)
                    sum = sum + torch.abs(output - label)/2

                print(sum/length)


def test(root, model):
    for

def get_class(labels):
    labels = labels[-3:]
    return torch.eye(250)[int(labels)-1,:].float()

def main(root):
    voice = []
    label = []
    for labels in os.listdir(root):
        if labels == ".DS_Store":
            continue
        for voices in os.listdir(os.path.join(root,labels)):
            if voices == ".DS_Store":
                continue
            if is_voice_file(voices):
                path = os.path.join(root,labels,voices)
                voice.append(path)
                label.append(get_class(labels))

    print(len(voice))
    print(len(label))
    kfold = KFold(n_splits=5,shuffle=True)
###################################
    batch_size = 64
    workers = 4
###################################
    for i, (train_idx, val_idx) in enumerate(kfold.split(voice, label)):
        trainset, valset = np.array(voice)[[train_idx]], np.array(voice)[[val_idx]]
        traintag, valtag = np.array(label)[[train_idx]], np.array(label)[[val_idx]]
        train_dataset = MyDataset(trainset, traintag)
        val_dataset = MyDataset(valset, valtag)



if __name__ == "__main__":
    root = "/Users/wangrui/Downloads/LibriSpeech-SI/train"
    main(root)


