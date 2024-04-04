from training_functions import pretrain, get_ucr_data, get_parameters
from evaluation import evaluate_model_split
from preprocessing import get_train_val, PrematureDatasetSplit
from model import multichannelLSTM, Identity

import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn

def loop_source_data():
    #TODO: make function that loops source data sets to pretrain on
    
    pass

def split_features(params):
    data = pd.read_csv("./data/train_test_data/trainval.csv", index_col=0)
    train, val = get_train_val(data)  
    
    dataset = PrematureDatasetSplit(train)
    dataloader = DataLoader(dataset, batch_size = int(params["batch_size"]), shuffle = True)
    
    valset = PrematureDatasetSplit(val)
    valloader = DataLoader(valset, batch_size = int(params["batch_size"]), shuffle = True)

    return dataloader, valloader

def multichannel_finetune():
    """_summary_

    Args:
        train (_type_): _description_
        model (_type_): _description_
    """
    params = get_parameters("./hyperparameter_testing/parameter_testing_tl_29:03:2024_00:33:14.txt")

    source_train, source_val, target_size = get_ucr_data("/Users/yarianrango/Documents/School/Master-AI-VU/Thesis/data/UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv", 0.3, params)
    train_loss, val_loss, val, model = pretrain(source_train, source_val, target_size, params)
    
    # Freeze LSTM layers and remove final linear layer
    for param in model.parameters():
        param.requires_grad = False
        
    model._modules["lin1"] = Identity()
    
    # Get dataloader
    train, val = split_features(params)
    
    # Init multichannel model
    mLSTM = multichannelLSTM(model, params)
    
    # Get loss function and optimizer
    loss_function = nn.BCEWithLogitsLoss()
    device = "cpu"
    
    optimizer = getattr(optim, params['optimizer_fine'])(mLSTM.parameters(), lr= params['learning_rate_fine'])
    
    val_loss_list = []
    train_loss = []
    for epoch in range(int(params["epochs_fine"])):
        mLSTM.train()
        total_loss = 0.0
    
        val_loss = 0
        for (x1, x2, x3), labels in train:
            mLSTM.zero_grad()

            output = mLSTM(x1.unsqueeze(-1), x2.unsqueeze(-1), x3.unsqueeze(-1))
            
            label_correct = labels.unsqueeze(-1).float().to(device)

            loss = loss_function(output, label_correct)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        avg_loss = total_loss/len(train)
        train_loss.append(avg_loss)
        
        model.eval()
        with torch.no_grad():
            for (x1, x2, x3), labels in val:
                output = mLSTM(x1.unsqueeze(-1), x2.unsqueeze(-1), x3.unsqueeze(-1))
            
                label_correct = labels.unsqueeze(-1).float().to(device)

                vloss = loss_function(output, label_correct)
        
                val_loss += vloss.item()
            avg_val_loss = val_loss/len(val)
            val_loss_list.append(avg_val_loss)
            
    return train_loss, val_loss_list, val, mLSTM
    
train_loss, val_loss_list, val, mLSTM = multichannel_finetune()

evaluate_model_split(train_loss, val_loss_list, val, mLSTM, plot = True)
    
    
