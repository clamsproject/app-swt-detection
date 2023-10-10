import torch
import torch.nn as nn
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import argparse
from d2l import torch as d2l
import time
from tempfile import TemporaryDirectory


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
        image = read_image(str(self.images[image_name])).float()
        label = get_label(self.table.iloc[index]['type label'])
        return (image, label)
    
def get_label(label):
    slate = ["S"]
    chyron = ["I","N","Y"]
    credit = ["C"]
    if label in slate:
        return 0
    elif label in chyron:
        return 1
    elif label in credit:
        return 2
    else:
        return 3

def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

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
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)      
    train_size = len(dataset) 
    class_names =  ['slate', 'chyron', 'credit', 'not-interested']                
    loss = nn.CrossEntropyLoss(reduction="none")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model(get_net(), train_loader, train_size, loss, device)

def train_model(model, train_loader, train_size, criterion, device, num_epochs=25):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = Path(tempdir) / 'best_model_params.pt'

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        loss.sum().backward()

                running_loss += loss.sum().item() * inputs.size(0)
                print(preds)
                print(labels.data)
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / train_size
                epoch_acc = running_corrects.double() / train_size

                print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path))
    return model

                                                           
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", help="root directory containing the images to train on")
    args = parser.parse_args()
    dataset = build_dataset(args.root_directory)
    train(dataset)

main()