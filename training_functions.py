import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
import os

from model import LSTM, CNN
from preprocessing import UCRDataset, PrematureDataset, train_val_split
from sklearn.model_selection import StratifiedKFold
from evaluation import evaluate_model


def get_ucr_data(data_dir, test_size, params):
    dataset = UCRDataset(data_dir)
    target_size = dataset.target_size
    
    train, val = train_val_split(dataset, test_size)
    
    train_loader = DataLoader(train, batch_size = int(params["batch_size"]), shuffle=True)
    val_loader = DataLoader(val ,batch_size = int(params["batch_size"]), shuffle=True)
    
    return train_loader, val_loader, target_size

def get_preterm_data(test_size, batch_size):
    dataset = PrematureDataset("./data/train_test_data/trainval.csv", 1)

    train, val = train_val_split(dataset, test_size)
    
    print(train.dataset)
    train_loader = DataLoader(train, batch_size, shuffle=True)
    val_loader = DataLoader(val ,batch_size, shuffle=True)
    
    return train_loader, val_loader

def get_parameters(path):
    with open(path, "r") as file:
        x = file.read()
    x = x.replace(":", "")
    x = x.split(sep = " ")
    params = {}
    for i in range(len(x)-1):
        if i % 2 == 0:
            params[x[i]] = x[i + 1]
            if not params[x[i]].isalpha():
                params[x[i]] = float(params[x[i]])
    return params

def train_wo_transfer_learning(params, test_size):
    train, val, test = get_preterm_data(test_size, params["batch_size"])

    model = LSTM(params)
    
    pos_weight = torch.tensor([7])
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    device = "cpu"
    
    optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=round(params["epochs"]*0.8))

    train_loss = []
    val_loss_list = []
    model.to(device)

    for epoch in tqdm(range(params["epochs"])):
        model.train()
        total_loss = 0.0
    
        val_loss = 0
        for sequence, label in train: 
            model.zero_grad()
            
            #TODO: Fix input shape independent of data set
            sequence_shaped = sequence.float().to(device)
            output = model(sequence_shaped)
            
            label_correct = label.unsqueeze(-1).float().to(device)

            loss = loss_function(output, label_correct)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss/len(train)
        train_loss.append(avg_loss)
        
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        # print("Epoch %d: Adam lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
        
        model.eval()
        with torch.no_grad():
            for sequence, label in val:
                sequence_shaped = sequence.float().to(device)
                output = model(sequence_shaped)
                
                label_correct = label.unsqueeze(-1).float().to(device)
                vloss = loss_function(output, label_correct)
        
                val_loss += vloss.item()
            avg_val_loss = val_loss/len(val)
            val_loss_list.append(avg_val_loss)
        
    return train_loss, val_loss_list, val, model

def pretrain(model, train, val, target_size, params, device, binary_classes = False):

    if binary_classes:
        loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = nn.CrossEntropyLoss()
    
    optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=round(params["epochs"]*0.8))
    
    train_loss = []
    val_loss_list = []
    model.to(device)

    for epoch in tqdm(range(int(params["epochs"]))):
        model.train()
        total_loss = 0.0
    
        val_loss = 0
        for sequence, label in train: 

            model.zero_grad()
            
            #TODO: Fix input shape independent of data set
            sequence_shaped = sequence.unsqueeze(-1).to(torch.float32).to(device).permute(0,2,1)
            output = model(sequence_shaped).squeeze(-1)
            if binary_classes:
                label_correct = label.to(device).float()
            else:
                label_correct = label.to(device).long()
            loss = loss_function(output, label_correct)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss/len(train)
        train_loss.append(avg_loss)

        # scheduler.step()
        
        model.eval()
        with torch.no_grad():
            for sequence, label in val:
                sequence_shaped = sequence.unsqueeze(-1).to(torch.float32).to(device).permute(0,2,1)
                output = model(sequence_shaped).squeeze(-1)
                if binary_classes:
                    label_correct = label.to(device).float()
                else:
                    label_correct = label.to(device).long()
                vloss = loss_function(output, label_correct)
        
                val_loss += vloss.item()
            avg_val_loss = val_loss/len(val)
            val_loss_list.append(avg_val_loss)
        
    return train_loss, val_loss_list, val, model

def finetune(train, val, model, params, device):
    
    # Replace last layer with new layer
    for param in model.parameters():
        param.requires_grad = False
    
    model._modules["lin1"] = nn.Linear(int(params["hidden_dim"]) * 2, 1)
    
    # Train on pretrained model
    pos_weight = torch.tensor([5.5])
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = getattr(optim, params['optimizer_fine'])(model.parameters(), lr= params['learning_rate_fine'])
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=round(int(params["epochs_fine"])*0.8))

    train_loss = []
    val_loss_list = []
    model.to(device)

    for epoch in tqdm(range(int(params["epochs_fine"]))):
        model.train()
        total_loss = 0.0
    
        val_loss = 0
        for sequence, label in train: 

            model.zero_grad()
            
            #TODO: Fix input shape independent of data set
            sequence_shaped = sequence.unsqueeze(-1).float().to(device)

            output = model(sequence_shaped)
            
            label_correct = label.unsqueeze(-1).float().to(device)

            loss = loss_function(output, label_correct)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss/len(train)
        train_loss.append(avg_loss)
        
        # scheduler.step()
        
        model.eval()
        with torch.no_grad():
            for sequence, label in val:
                sequence_shaped = sequence.unsqueeze(-1).float().to(device)
                output = model(sequence_shaped)
                
                label_correct = label.unsqueeze(-1).float().to(device)
                vloss = loss_function(output, label_correct)
        
                val_loss += vloss.item()
            avg_val_loss = val_loss/len(val)
            val_loss_list.append(avg_val_loss)
        
    return train_loss, val_loss_list,  model

def perform_transfer_learning(source_data_dir, test_size, params):
    train, val, target_size = get_ucr_data(source_data_dir, test_size, params)
    
    train_loss, val_loss, val, model = pretrain(train, val, target_size, params)
    
    preterm_train, preterm_val = get_preterm_data(0.3, int(params["batch_size_fine"]))

    fine_train_loss, fine_val_loss,  finetuned_model = finetune(preterm_train, preterm_val, model, params)
    
    return fine_train_loss, fine_val_loss, preterm_val, finetuned_model

def overfit(params, train):
    model = LSTM(params)
    
    loss_function = nn.BCEWithLogitsLoss()
    device = "cpu"
    
    optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=3000)

    train_loss = []
    val_loss_list = []
    model.to(device)
    sequence, label = next(iter(train))
    sig = nn.Sigmoid()
    for epoch in tqdm(range(params["epochs"])):
        model.train()
        total_loss = 0.0
    
        val_loss = 0

        model.zero_grad()
            
            #TODO: Fix input shape independent of data set
        sequence_shaped = sequence.float().to(device)
        output = model(sequence_shaped)
            
        label_correct = label.unsqueeze(-1).float().to(device)
        pred = sig(output).round().int()
        print("output")
        print(output)
        print("pred")
        print(pred)
        print("label_correct")
        print(label_correct)
        print()
        loss = loss_function(output, label_correct)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_loss.append(total_loss)
        
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: Adam lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
    return train_loss, val_loss_list, model

def train_best_cnn(data_dir, params_dir, device = "cpu"):
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):  
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.tsv') and 'TRAIN' in file_name:
                    print(f"Filename: {file_name}:")
                    file_path = os.path.join(folder_path, file_name)
                    
                    for params_name in os.listdir(params_dir):
                        if file_name in params_name:
                            print(f"Params: {params_name}")
                            params_path = os.path.join(params_dir, params_name)
                            params = get_parameters(params_path)
                            source_train, source_val, target_size = get_ucr_data(file_path, 0.3, params)
                            if target_size == 2:
                                target_size = 1  
                                model = CNN(target_size)
                                train_loss, val_loss, val, model = pretrain(model, source_train, source_val, target_size, params, device, binary_classes=True)
        
                            else:
                                model = CNN(target_size)
                                train_loss, val_loss, val, model = pretrain(model, source_train, source_val, target_size, params, device)
                            model_path = f"./models/pretrained_cnns/{file_name}.pt"
                            torch.save(model.state_dict(), model_path)
  
def kfold_sDNN(model_name, target_size, params_fine, device = "cpu"):
    if target_size == 2:
        target_size = 1
    
    if model_name == "CNN":
        model = CNN(target_size, input_dim=3)
    
    dataset = PrematureDataset("./data/train_test_data/trainval.csv", n_components=3)
    
    k=5
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)

    # Get loss function and optimizer
    pos_weight = torch.tensor([8.7]).to(device) # based on ratio of data class imbalance
    loss_function = nn.BCEWithLogitsLoss(pos_weight)
    
    optimizer = getattr(optim, params_fine['optimizer'])(model.parameters(), lr= params_fine['learning_rate'])
    
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
            
        auc = evaluate_model(train_loss_list, val_loss_list, val_loader, model)
        print(f"AUC of {auc} for fold {fold+1}")
        tot_auc += auc
    avg_auc = tot_auc/k
    
    return avg_auc   



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
                
# train_best_cnn(data_dir="../data/source_datasets/", params_dir="./hyperparameter_testing/cnn_source_data/")
# params_pre = get_parameters("./hyperparameter_testing/parameter_testing_pretrain_05:04:2024_21:52:26.txt")
# params_pre["epochs"] = 40

# source_train, source_val, target_size = get_ucr_data("/Users/yarianrango/Documents/School/Master-AI-VU/Thesis/data/test/ECG5000/ECG5000_TEST.tsv", 0.3, params_pre)
# model = CNN(target_size=target_size)
# train_loss, val_loss, val, model = pretrain(model, source_train, source_val, target_size, params_pre, device = "cpu", binary_classes=False)

# from evaluation import evaluate_multiclass
# evaluate_multiclass(train_loss, val_loss, val, model, plot = True)

# torch.save(model.state_dict(), "./models/pretrained_cnn")