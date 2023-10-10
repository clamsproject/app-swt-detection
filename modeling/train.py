import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from skimage import io
import argparse
import matplotlib.pyplot as plt
import numpy as np


label_keys = ["S","I","N","Y","C","B","W","L","O","M","E","K","G","T","F","R"]

class FrameDetectionDataset(Dataset):
    def __init__(self, table, images):
        self.table = table
        self.images = images

    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image_name = self.table.iloc[index]['filename']
        image = io.imread(self.images[image_name])
        return (TF.to_tensor(image), self.table.iloc[index]['type label'])
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def build_dataset(directory):
    p = Path(directory)
    csvs = list(p.glob('**/*.csv'))
    images = list(p.glob('**/*.jpg'))
    image_names = {image.name:image for image in images}
    tables = []
    for csv in csvs:
        tables.append(pd.read_csv(csv))
    table = pd.concat(tables)
    table = table.loc[table['filename'].isin(image_names)]
    return FrameDetectionDataset(table, image_names)

def train(dataset):
    slate = ["S"]
    chyron = ["I","N","Y"]
    credit = ["C"]
    not_interested = ["B","W","L","O","M","E","K","G","T","F","R"]
    classes = ('slate', 'chyron', 'credit', 'not-interested')
    trainloader = DataLoader(dataset, shuffle=True)
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                                                               
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", help="root directory containing the images to train on")
    args = parser.parse_args()
    dataset = build_dataset(args.root_directory)
    train(dataset)

main()