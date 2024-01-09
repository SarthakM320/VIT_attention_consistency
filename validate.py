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
import argparse

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

def main(args):
    csv = pd.read_csv('AID_data_val.csv')
    device = args['device']
    exp_name = 'Experiments_new/'+args['exp_name']
    # exp_name = 'Experiments\Experiments_colab\\baseline_all_three_seed_16'
    # exp_name = 'Experiments\Experiments_colab\lora_all_three_seed_16'
    # exp_name = 'Experiments\\baseline_adam_lr0.005_imagenet'

    

    data = dataset(csv)
    dataloader = DataLoader(data, shuffle = True, batch_size=32)
    
    if args['model_type'] == 'base_imagenet':
        model = Model()
    elif args['model_type'] == 'lora':
        model = LORAModel()
    elif args['model_type'] == 'lora_mod':
        model = LORAModelMod()
    

    epoch = model.load_model(exp_name, latest = False)

    model = model.to(device)
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
    parser = argparse.ArgumentParser(description="Argument Parser for your script")
    parser.add_argument('--model_type', choices=['base_imagenet', 'lora', 'lora_mod'], help='Model type', default = 'base_imagenet')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--device', type=str, default='cuda', help='Device (e.g., cuda or cpu)')
    parser.add_argument('--seed', type=int, default=16, help='Random seed')
    args = vars(parser.parse_args())
    main(args)


    