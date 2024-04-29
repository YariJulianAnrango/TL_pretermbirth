from training_functions import pretrain, get_ucr_data, get_parameters
from evaluation import evaluate_model_split
from preprocessing import get_train_val, PrematureDatasetSplit
from model import MultiChannelModel, Identity, LSTM, CNN

import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np

def loop_source_data():
    #TODO: make function that loops source data sets to pretrain on
    
    pass

def split_features(params, val_split = True):
    data = pd.read_csv("./data/train_test_data/trainval.csv", index_col=0)
    
    if not val_split:
        dataset = PrematureDatasetSplit(data)
        
        return dataset
    
    train, val = get_train_val(data)  
    
    dataset = PrematureDatasetSplit(train)
    dataloader = DataLoader(dataset, batch_size = int(params["batch_size"]), shuffle = True)
    
    valset = PrematureDatasetSplit(val)
    valloader = DataLoader(valset, batch_size = int(params["batch_size"]), shuffle = True)
    
    return dataloader, valloader

def train_epoch(model, train_loader, criterion, optimizer, device):
    total_loss = 0.0

    for x, labels in train_loader:
        model.zero_grad()
        
        output = model(x)
            
        label_correct = labels.unsqueeze(-1).to(torch.float32).to(device)

        loss = criterion(output, label_correct)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
            
    avg_loss = total_loss/len(train_loader)
    
    return avg_loss

def valid_epoch(model, val_loader, criterion, device):
    val_loss = 0
    
    with torch.no_grad():
        for x, labels in val_loader:
            output = model(x)
            
            label_correct = labels.unsqueeze(-1).to(torch.float32).to(device)

            vloss = criterion(output, label_correct)
        
            val_loss += vloss.item()
        avg_val_loss = val_loss/len(val_loader)
        
    return avg_val_loss

def multichannel_finetune(model_name, source_path, params_fine = None, params_pre = None, device = "cpu"):
    """Trains multichannel concatenation model

    Args:
        model: string indicating the use of the LSTM or CNN
        source_path: string indicating path to source dataset
        params_fine: dictionary containing parameters for funetining
        params_pre: dictionary containing parameters for pretraining. Will default to best parameters if none is given
        device: pytorch device, defaults to cpu
    """
    source_train, source_val, target_size = get_ucr_data(source_path, 0.3, params_pre)
    
    # Get multimodel
    multimodel = get_multichannelmodel(model_name, params_pre, params_fine, target_size, device)
    
    # Get dataloader
    train, val = split_features(params_fine)
    
    # Get loss function and optimizer
    pos_weight = torch.tensor([params_fine["loss_weight"]]).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight)
    
    optimizer = getattr(optim, params_fine['optimizer'])(multimodel.parameters(), lr= params_fine['learning_rate'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=round(int(params_fine["epochs"])*0.8))
    
    val_loss_list = []
    train_loss_list = []
 
    # Training loop
    for epoch in tqdm(range(int(params_fine["epochs"]))):
        multimodel.train()
        train_loss = train_epoch(multimodel, train, loss_function, optimizer, device)
        train_loss_list.append(train_loss)
        
        scheduler.step()
        
        # Evaluate in between epochs
        multimodel.eval()
        val_loss = valid_epoch(multimodel, val, loss_function, device)
        val_loss_list.append(val_loss)
            
    return train_loss_list, val_loss_list, val, multimodel

def kfold_multichannel(model_name, model_path, target_size, params_fine = None, params_pre = None, device = "cpu"):
     
    # Get multichannel model    
    if target_size == 2:
        target_size = 1
        
    model = get_multichannelmodel(model_name, model_path, params_fine,target_size, device, params_pre)
    
    # Get dataloader
    dataset = split_features(params_fine, val_split=False)
    
    # Init Kfold 
    k=5
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)

    # Get loss function and optimizer
    pos_weight = torch.tensor([8.7]).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight)
    
    optimizer = getattr(optim, params_fine['optimizer'])(model.parameters(), lr= params_fine['learning_rate'])
    # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=round(int(params_fine["epochs"])*0.8))
 
    tot_auc = 0
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(dataset.X, dataset.y)):
        val_loss_list = []
        train_loss = []
        print('Fold {}'.format(fold + 1))
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=params_fine["batch_size"], sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=params_fine["batch_size"], sampler=val_sampler)
        
        # Training loop
        train_loss_list = []
        val_loss_list = []
        
        for epoch in tqdm(range(int(params_fine["epochs"]))):
            model.train()
            train_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
            train_loss_list.append(train_loss)
            
            # scheduler.step()
        
            # Evaluate in between epochs
            model.eval()
            val_loss = valid_epoch(model, val_loader, loss_function, device)
            val_loss_list.append(val_loss)
            
        auc = evaluate_model_split(train_loss_list, val_loss_list, val_loader, model, device = "cpu")
        print(f"AUC of {auc} for fold {fold+1}")
        tot_auc += auc
    avg_auc = tot_auc/k
    
    return avg_auc

def get_multichannelmodel(model_name, model_path, params_fine, target_size, device, params_pre = None):
    if model_name == "LSTM":
        model = load_pretrained_lstm(model_path, params_pre, target_size, input_dim = 1, device = device)
        hidden_dim = params_pre["hidden_dim"]
    elif model_name == "CNN":
        model = load_pretrained_cnn(model_path, target_size, device)
        hidden_dim = 128
        
    for param in model.parameters():
        param.requires_grad = False
        
    model._modules["lin1"] = Identity()
    
    multimodel = MultiChannelModel(model, params_fine, hidden_dim=hidden_dim)
    multimodel.to(device)
    
    return multimodel

def load_pretrained_lstm(path, params, target_size, input_dim = 1, device = "cpu"):
    model = LSTM(params, target_size, input_dim, device)
    model.load_state_dict(torch.load(path))
    
    return model

def load_pretrained_cnn(path, target_size, device = "cpu"):
    model = CNN(target_size)
    model.load_state_dict(torch.load(path))
    
    return model
# params = {
#         "batch_size": 10,
#         "learning_rate": 0.01,
#         "optimizer": "Adam",
#         "epochs": 1,
#         "lin_layers": 2,
#         "hidden1": 4,
#         "hidden2": 4
#     }
# auc = kfold_multichannel("CNN", "./models/pretrained_cnns/ECG5000_TRAIN.tsv.pt", 5, params_fine = params, device = "cpu", params_pre=None)
# params_fine = get_parameters("./hyperparameter_testing/parameter_testing_mLSTM_kfold_09:04:2024_22:12:09.txt")

# train_loss, val_loss_list, val, mLSTM = multichannel_finetune(params_fine=params_fine)

# evaluate_model_split(train_loss, val_loss_list, val, mLSTM, plot = True)

# load_pretrained_cnn("/Users/yarianrango/Documents/School/Master-AI-VU/Thesis/TL_pretermbirth/models/pretrained_cnns/NonInvasiveFetalECGThorax2_TRAIN.pt", target_size=42)
