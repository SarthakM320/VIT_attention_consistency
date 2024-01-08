from model import LORAModel, Model, LORAModelMod
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn 
import pandas as pd
from torchvision import transforms
import logging
import sys
from tqdm import tqdm 
from PIL import Image
import os

class dataset(Dataset):
    def __init__(self, df):
        self.images = df['image'].values
        self.labels = df['label'].values
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.images[idx])), self.labels[idx]

def main():
    csv = pd.read_csv('AID_data_val.csv')
    # exp_name = 'Experiments\Experiments_colab\\baseline_all_three_seed_16'
    exp_name = 'Experiments\Experiments_colab\lora_all_three_seed_16'
    # exp_name = 'Experiments\\baseline_adam_lr0.005_imagenet'

    if 'lora' in exp_name:
        epoch = int(os.listdir(exp_name)[-1].split('.')[0].split('_')[-1])
    else:
        epoch = int(os.listdir(exp_name)[-2].split('.')[0].split('_')[-1])
    

    data = dataset(csv)
    dataloader = DataLoader(data, shuffle = True, batch_size=32)
    
    if 'baseline' in exp_name:
        model = Model()
    elif 'lora' in exp_name and 'mod' in exp_name:
        model = LORAModelMod()
    else:
        model = LORAModel()

    model.load_model(exp_name, epoch)
    model = model.cuda()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx,data in enumerate(tqdm(dataloader)):
            image, label = data
            image = image.cuda()
            label = label.cuda()

            _,predictions,_,_ = model(image,torch.zeros_like(image))
            predictions = predictions.softmax(dim = -1).argmax(dim = -1)
            total += len(label)
            correct += sum(predictions == label)

    print('Accuracy: ', (correct/total).item()*100)

if __name__ == "__main__":
    main()


    