import torch
from torch import nn
import numpy as np
import random
from model import Model, LORAModel
from utils_own import horizontal_flip, horizontal_flip_target, vertical_flip, vertical_flip_target
from data import train_dataset, val_dataset, triplet_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed = 2**4
    set_seed(seed)
    seed = 2**4
    set_seed(seed)
    # exp = 'Experiments/baseline_adam_lr0.005_dino_imagenet'
    exp=f'Experiments/lora_mod_without_attnloss_seed_{seed}'
    # trial_4 was changing the attention weights also with attention loss
    # trial 5 was not changing the attention weights with attention loss with a loss weight of 10^3
    # trial 6 was changing only the qkv values in attention blocks with attention loss with a loss weight of 10^3
    # trial 7 was not changing the attention weights with attention loss with a loss weight of 10^4
    # lora - lora adaptation with attention loss of weight 10^4 (lora_2 is rerun)
    # lora_mod - lora adaptation modified with attention loss of weight 10^4 (lora_mod_2 is rerun)
    writer = SummaryWriter(exp)
    num_epochs = 15
    device = 'cuda'
    # model = Model().to(device)
    model = LORAModel().to(device)
    triplet_data = triplet_dataset()
    triplet_val_data = triplet_dataset(csv_path='AID_data_triplet_val.csv')

    train_dataloader = DataLoader(triplet_data, batch_size = 32, shuffle = True)
    val_dataloader = DataLoader(triplet_val_data, batch_size = 32, shuffle = True)

    loss_output_fn = nn.CrossEntropyLoss()
    loss_attn_fn = nn.MSELoss()
    loss_triplet_fn = nn.TripletMarginLoss()

    optim = torch.optim.Adam(model.parameters(), lr = 0.005)
    step_train = 0
    step_val = 0
    resume = False
    resume_epochs = 0
    attn_loss_weight = 10**4

    if resume:
        model.load_model(f'{exp}', resume_epochs)
        model = model.to(device)

    for epoch in range(resume_epochs+1, num_epochs+1):
        print(f'EPOCH {epoch}')
        overall_output_1 = []
        overall_output_2 = []
        overall_labels = []  

        model.train()
        for idx, input_data in enumerate(tqdm(train_dataloader)):
            anc, image_2, labels, type, pos, neg = input_data
            anc = anc.to(device)
            image_2 = image_2.to(device)
            labels = labels.to(device)
            b = len(labels)
            pos = pos.to(device)
            neg = neg.to(device)

            
            self_attn_1, output_1, self_attn_2, output_2 = model(anc, image_2)

            frames = []
            for i in range(b):
                frames.append(flip(self_attn_1[i],type[i]).unsqueeze(0))

            self_attn_1 = torch.cat(frames)
            
            loss_out = (loss_output_fn(output_1, labels) + loss_output_fn(output_2, labels)) # TODO try weighted cross entropy also
            loss_att = loss_attn_fn(self_attn_1, self_attn_2)*attn_loss_weight 

            anc_features, pos_features, neg_features = model.triplet_forward(anc, pos, neg)
            loss_trip = loss_triplet_fn(anc_features, pos_features, neg_features)*10
            # print(f'ANC: {anc_features.shape}')
            # print(f'ANC: {pos_features.shape}')
            # print(f'ANC: {neg_features.shape}')
            # print(f'Loss: {loss_trip}')
            loss = loss_out + loss_att + loss_trip
            # loss = loss_out
            optim.zero_grad()
            loss.backward()
            optim.step()

            overall_output_1.append(output_1)
            overall_output_2.append(output_2)   
            overall_labels.append(labels)
            
            if step_train%1 == 0:
                writer.add_scalar('Loss_overall/train', loss.item(), step_train)
                writer.add_scalar('Loss_output/train', loss_out.item(), step_train)
                writer.add_scalar('Loss_attention/train', loss_att.item(), step_train)
                writer.add_scalar('Loss_triplet/train', loss_trip.item(), step_train)


            # if step_train%10 == 0:
                # print(f'Epoch: {epoch}, Loss Overall: {loss.item()}, Loss Output: {loss_out}, Loss attention: {loss_att}')
                # precision, recall, f1, acc = get_results(torch.cat(overall_output_1), torch.cat(overall_labels),verbose=False)
                # writer.add_scalar('Output_1/precision_train', precision, step_train)
                # writer.add_scalar('Output_1/recall_train', recall, step_train)
                # writer.add_scalar('Output_1/f1_train', f1, step_train)
                # writer.add_scalar('Output_1/accuracy_train', acc, step_train)

                # precision, recall, f1, acc = get_results(torch.cat(overall_output_2), torch.cat(overall_labels),verbose=False)
                # writer.add_scalar('Output_2/precision_train', precision, step_train)
                # writer.add_scalar('Output_2/recall_train', recall, step_train)
                # writer.add_scalar('Output_2/f1_train', f1, step_train)
                # writer.add_scalar('Output_2/accuracy_train', acc, step_train)

            step_train += 1

        precision, recall, f1, acc = get_results(torch.cat(overall_output_1), torch.cat(overall_labels))
        writer.add_scalar('Output_1/precision_train', precision, epoch)
        writer.add_scalar('Output_1/recall_train', recall, epoch)
        writer.add_scalar('Output_1/f1_train', f1, epoch)
        writer.add_scalar('Output_1/accuracy_train', acc, epoch)

        precision, recall, f1, acc = get_results(torch.cat(overall_output_2), torch.cat(overall_labels))
        writer.add_scalar('Output_2/precision_train', precision, epoch)
        writer.add_scalar('Output_2/recall_train', recall, epoch)
        writer.add_scalar('Output_2/f1_train', f1, epoch)
        writer.add_scalar('Output_2/accuracy_train', acc, epoch)
        
        model.save_model(epoch,f'{exp}')
               


        overall_output_1 = []
        overall_output_2 = []
        overall_labels = []   
        
        model.eval()
        for idx, input_data in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():
                anc, image_2, labels, type, pos, neg = input_data
                anc = anc.to(device)
                image_2 = image_2.to(device)
                labels = labels.to(device)
                b = len(labels)
                pos = pos.to(device)
                neg = neg.to(device)

                self_attn_1, output_1, self_attn_2, output_2 = model(anc, image_2)

                frames = []
                for i in range(b):
                    frames.append(flip(self_attn_1[i],type[i]).unsqueeze(0))
                self_attn_1 = torch.cat(frames)
                output_1 = output_1.softmax(dim=-1)
                output_2 = output_2.softmax(dim=-1)
                loss_out = loss_output_fn(output_1, labels) + loss_output_fn(output_2, labels) # TODO try weighted cross entropy also
                loss_att = loss_attn_fn(self_attn_1, self_attn_2)* attn_loss_weight

                anc_features, pos_features, neg_features = model.triplet_forward(anc, pos, neg)
                loss_trip = loss_triplet_fn(anc_features, pos_features, neg_features)*10        
                loss = loss_out + loss_att

                if step_val%5:
                    writer.add_scalar('Loss_overall/val', loss.item(), step_val)
                    writer.add_scalar('Loss_output/val', loss_out.item(), step_val)
                    writer.add_scalar('Loss_attention/val', loss_att.item(), step_val)
                    writer.add_scalar('Loss_triplet/val', loss_trip.item(), step_val)


                step_val+=1
                overall_output_1.append(output_1)
                overall_output_2.append(output_2)
                overall_labels.append(labels)
        
       
        
        print('OUTPUT_1')
        precision, recall, f1, acc = get_results(torch.cat(overall_output_1), torch.cat(overall_labels))
        writer.add_scalar('Output_1/precision_val', precision, epoch)
        writer.add_scalar('Output_1/recall_val', recall, epoch)
        writer.add_scalar('Output_1/f1_val', f1, epoch)
        writer.add_scalar('Output_1/accuracy_val', acc, epoch)
        print()
        print('OUTPUT_2')
        precision, recall, f1, acc = get_results(torch.cat(overall_output_2), torch.cat(overall_labels))
        writer.add_scalar('Output_2/precision_val', precision, epoch)
        writer.add_scalar('Output_2/recall_val', recall, epoch)
        writer.add_scalar('Output_2/f1_val', f1, epoch)
        writer.add_scalar('Output_2/accuracy_val', acc, epoch)

        step_val+=1
                 



def flip(x, type):
    if type == 'horizontal':
        return horizontal_flip_target(horizontal_flip(x))
    else:
        return vertical_flip_target(vertical_flip(x))


def get_acc(y_true, y_pred):
    return sum(y_true == y_pred)/len(y_true)

def get_results(output, labels, verbose = True):
    y_true = labels.cpu().detach().numpy()
    y_pred = output.softmax(dim=-1).argmax(dim = -1).cpu().detach().numpy()
    precision = precision_score(y_true=y_true, y_pred=y_pred,average="macro")
    recall = recall_score(y_true=y_true, y_pred=y_pred,average="macro")
    f1 = f1_score(y_true=y_true, y_pred=y_pred,average="macro")
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    # acc = get_acc(y_true=y_true, y_pred=y_pred)
    
    if verbose:
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'Accuracy: {acc}')
    return precision, recall, f1, acc

if __name__ == "__main__":
    main()

