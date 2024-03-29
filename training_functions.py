import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import pandas as pd
from sklearn import metrics

import matplotlib.pyplot as plt
from tqdm import tqdm

from model import LSTM
from preprocessing import UCRDataset, PrematureDataset, train_val_split


def get_ucr_data(data_dir, test_size, params):
    dataset = UCRDataset(data_dir)
    target_size = dataset.target_size
    
    train, val = train_val_split(dataset, test_size)
    
    train_loader = DataLoader(train, batch_size = params["batch_size"], shuffle=True)
    val_loader = DataLoader(val ,batch_size = params["batch_size"], shuffle=True)
    
    return train_loader, val_loader, target_size

def get_preterm_data(test_size, batch_size):
    dataset = PrematureDataset("./data/train_test_data/trainval.csv", 1)

    train, val = train_val_split(dataset, test_size)
    
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

def pretrain(train, val, target_size, params):

    model = LSTM(params, target_size, input_dim=1)

    loss_function = nn.CrossEntropyLoss()
    device = "cpu"
    
    optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=round(params["epochs"]*0.8))
    
    train_loss = []
    val_loss_list = []
    model.to(device)

    for epoch in tqdm(range(int(500))):
        model.train()
        total_loss = 0.0
    
        val_loss = 0
        for sequence, label in train: 
            model.zero_grad()
            
            #TODO: Fix input shape independent of data set
            sequence_shaped = sequence.unsqueeze(-1).float().to(device)
                
            output = model(sequence_shaped)
            label_correct = label.to(device)

            loss = loss_function(output, label_correct)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss/len(train)
        train_loss.append(avg_loss)

        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            for sequence, label in val:
                sequence_shaped = sequence.unsqueeze(-1).float().to(device)
                output = model(sequence_shaped)
                label_correct = label.to(device)
                vloss = loss_function(output, label_correct)
        
                val_loss += vloss.item()
            avg_val_loss = val_loss/len(val)
            val_loss_list.append(avg_val_loss)
        
    return train_loss, val_loss_list, val, model

def finetune(train, val, model, params):
    
    # Replace last layer with new layer
    for param in model.parameters():
        param.requires_grad = False
    
    model._modules["lin1"] = nn.Linear(int(params["hidden_dim"]) * 2, 1)
    
    # Train on pretrained model
    pos_weight = torch.tensor([5.5])
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    device = "cpu"
    
    optimizer = getattr(optim, params['optimizer_fine'])(model.parameters(), lr= params['learning_rate_fine'])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=round(int(params["epochs_fine"])*0.8))

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
            sequence_shaped = sequence.float().to(device)
            output = model(sequence_shaped)
            
            label_correct = label.unsqueeze(-1).float().to(device)

            loss = loss_function(output, label_correct)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss/len(train)
        train_loss.append(avg_loss)
        
        scheduler.step()
        
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
        
    return train_loss, val_loss_list,  model
    
def evaluate_model(train_loss, val_loss, val, model, plot = False):
    if plot:
        plt.plot(train_loss, label = 'train loss')
        plt.plot(val_loss, label = 'val loss')
        plt.title("Train loss and val loss per epoch")
        plt.legend()
        plt.savefig("./figures/pca_results.png")
        plt.show()

    model.eval()
    preds = []
    labels = []

    sig = nn.Sigmoid()
    for sequence, label in val:
        sequence_shaped = sequence.float().to("cpu")
    
        logits_output = model(sequence_shaped)

        pred = sig(logits_output).round().int()
        print("pred",pred)
        print("label",label)
        for p in pred:
            preds.append(p.item())
        for l in label:
            labels.append(l.item())
        
    accuracy = metrics.accuracy_score(labels, preds)
    f1_score = metrics.f1_score(labels, preds)

    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    if plot:
        plt.plot(fpr, tpr)
        plt.title(f"ROC plot with AUC {auc}, accuracy {accuracy}, and f1 score {f1_score}")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.savefig("./figures/pca_roc.png")
        plt.show()
        print("f1_score", f1_score)
        print("accuracy", accuracy)
        print("auc", auc)
    
    return auc
    
def evaluate_multiclass(train_loss, val_loss, val, model, plot = False):
    if plot:
        plt.plot(train_loss, label = 'train loss')
        plt.plot(val_loss, label = 'val loss')
        plt.title("Train loss and val loss per epoch")
        plt.legend()
        plt.savefig("./figures/pca_results.png")
        plt.show()

    model.eval()
    preds = []
    labels = []

    softmax = nn.Softmax()
    for sequence, label in val:
        sequence_shaped = sequence.unsqueeze(-1).float().to("cpu")
    
        logits_output = model(sequence_shaped)
        # print("logits \n",logits_output)
        probs = softmax(logits_output)
        # print("probs \n",probs)
        pred = torch.argmax(probs, axis = 1)
        # print("pred \n",pred)
        # print("label \n",label)
        # print()
        for p in pred:
            preds.append(p.item())
        for l in label:
            labels.append(l.item())
        
    accuracy = metrics.accuracy_score(labels, preds)
    f1_score = metrics.f1_score(labels, preds, average="weighted")

    if plot:
        print(f"accuracy: {accuracy}, f1_score {f1_score}")
        
    return f1_score
    
def perform_transfer_learning(source_data_dir, test_size, params):
    train, val, target_size = get_ucr_data(source_data_dir, test_size, params)
    
    train_loss, val_loss, val, model = pretrain(train, val, target_size, params)
    
    preterm_train, preterm_val = get_preterm_data(0.3, int(params["batch_size_fine"]))

    fine_train_loss, fine_val_loss,  finetuned_model = finetune(preterm_train, preterm_val, model, params)
    
    return fine_train_loss, fine_val_loss, preterm_val, finetuned_model

def visualize_seq(data):
    for sequence, label in data: 
        print(sequence[0])
        print(label[0])
    
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

# print("Performing training without transfer learning")
# train_loss, val_loss_list, val, model = train_wo_transfer_learning(0.3)
# evaluate_model(train_loss, val_loss_list, val, model)


# print()
# print("Performing training with transfer learning")
# train_loss, val_loss, val, test, finetuned_model = perform_transfer_learning("/Users/yarianrango/Documents/School/Master-AI-VU/Thesis/data/UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv", 0.3)
# evaluate_model(train_loss, val_loss, val, finetuned_model, plot = True)

# train, val, test, target_size = get_ucr_data("/Users/yarianrango/Documents/School/Master-AI-VU/Thesis/data/UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv", 0.3)

# # visualize_seq(train)
# train_loss, val_loss, model = pretrain(train, val, target_size)
# train, val, test = get_preterm_data(0.3, 2)

# evaluate_multiclass(train_loss, val_loss, val, model)

# overfit(train)


# params = {"batch_size": 2,
#               "learning_rate": 0.1,
#               "optimizer": "Adam",
#               "layer_dim": 1,
#               "hidden_dim": 15,
#               "dropout": 0.3,
#               "epochs": 200}

# train_loss, val_loss_list, val, model = train_wo_transfer_learning(params, 0.3)
# # # train_loss, val_loss_list, model = overfit(params, train)
# evaluate_model(train_loss, val_loss_list, val, model, plot = True)

# get_parameters("./hyperparameter_testing/parameter_testing_baseline_26:03:2024_13:19:38.txt")