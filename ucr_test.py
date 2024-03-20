import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
from sklearn import metrics

import matplotlib.pyplot as plt
from tqdm import tqdm

from model import LSTM
from preprocessing import UCRDataset, PrematureDataset, train_val_test_split

seed = 1

def get_ucr_data(data_dir, test_size):
    dataset = UCRDataset(data_dir)
    target_size = dataset.target_size
    
    train, val, test = train_val_test_split(dataset, test_size)

    train_loader = DataLoader(train, batch_size = 13, shuffle=True)
    val_loader = DataLoader(val ,batch_size = 13, shuffle=True)
    test_loader = DataLoader(test, batch_size = 13, shuffle = True)
    
    return train_loader, val_loader, test_loader, target_size

def get_preterm_data(data_dir, test_size):
    dataset = PrematureDataset("./data/total_df.csv", 1)

    train, val, test = train_val_test_split(dataset, 0.3)
    
    train_loader = DataLoader(train, 13, shuffle=True)
    val_loader = DataLoader(val ,13, shuffle=True)
    test_loader = DataLoader(test ,13, shuffle=True)
    
    return train_loader, val_loader, test_loader

def get_parameters():
    #TODO: make function that reads best parameters
    pass

def train_wo_transfer_learning(test_size):
    train, val, test = get_preterm_data("./data/total_df.csv", test_size)

    input_dim = 1
    hidden_dim = 15
    target_size = 1
    layer_dim = 5

    model = LSTM(input_dim, hidden_dim, target_size, layer_dim)
    
    loss_function = nn.BCEWithLogitsLoss()
    device = "cpu"
    
    optimizer = optim.SGD(model.parameters(), lr=0.00034442093623802093)

    train_loss = []
    val_loss_list = []
    model.to(device)

    for epoch in tqdm(range(1000)):
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

def pretrain(train, val, target_size):
    input_dim = 1
    hidden_dim = 15
    layer_dim = 5

    model = LSTM(input_dim, hidden_dim, target_size, layer_dim)

    loss_function = nn.CrossEntropyLoss()
    device = "cpu"
    
    optimizer = optim.SGD(model.parameters(), lr=0.00034442093623802093)

    train_loss = []
    val_loss_list = []
    model.to(device)

    for epoch in tqdm(range(1000)):
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
        
    return train_loss, val_loss_list, model

def finetune(train, val, model):
    
    # Replace last layer with new layer
    for param in model.parameters():
        param.requires_grad = False
    
    model._modules["lin1"] = nn.Linear(30, 1)
    
    # Train on pretrained model
    loss_function = nn.BCEWithLogitsLoss()
    device = "cpu"
    
    optimizer = optim.SGD(model.parameters(), lr=0.00034442093623802093)

    train_loss = []
    val_loss_list = []
    model.to(device)

    for epoch in tqdm(range(1000)):
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
        
    return train_loss, val_loss_list, model
    
def evaluate_model(train_loss, val_loss, val, model):
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
        print(logits_output)
        pred = sig(logits_output).round().int()
        
        for p in pred:
            preds.append(p.item())
        for l in label:
            labels.append(l.item())
        
    accuracy = metrics.accuracy_score(labels, preds)
    f1_score = metrics.f1_score(labels, preds)

    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.title(f"ROC plot with AUC {auc}, accuracy {accuracy}, and f1 score {f1_score}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig("./figures/pca_roc.png")
    plt.show()
    print("f1_score", f1_score)
    print("accuracy", accuracy)
    
def evaluate_multiclass(train_loss, val_loss, val, model):
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
        print(logits_output)
        probs = softmax(logits_output)
        print(probs)
        pred = torch.argmax(probs, axis = 1)
        print(pred)
        print()
        for p in pred:
            preds.append(p.item())
        for l in label:
            labels.append(l.item())
        
    accuracy = metrics.accuracy_score(labels, preds)
    f1_score = metrics.f1_score(labels, preds, average="weighted")

    print(f"accuracy: {accuracy}, f1_score {f1_score}")
    

def perform_transfer_learning(source_data_dir, test_size):
    train, val, test, target_size = get_ucr_data(source_data_dir, test_size)

    train_loss, val_loss, model = pretrain(train, val, target_size)
    
    preterm_train, preterm_val, preterm_test = get_preterm_data("./data/total_df.csv", test_size)

    fine_train_loss, fine_val_loss, finetuned_model = finetune(preterm_train, preterm_val, model)
    
    return fine_train_loss, fine_val_loss, preterm_val, preterm_test, finetuned_model

# print("Performing training without transfer learning")
# train_loss, val_loss_list, val, model = train_wo_transfer_learning(0.3)
# evaluate_model(train_loss, val_loss_list, val, model)

# print()
# print("Performing training with transfer learning")
# train_loss, val_loss, val, test, finetuned_model = perform_transfer_learning("/Users/yarianrango/Documents/School/Master-AI-VU/Thesis/data/UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv", 0.3)
# evaluate_model(train_loss, val_loss, val, finetuned_model)

train, val, test, target_size = get_ucr_data("/Users/yarianrango/Documents/School/Master-AI-VU/Thesis/data/UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv", 0.3)

train_loss, val_loss, model = pretrain(train, val, target_size)

evaluate_multiclass(train_loss, val_loss, val, model)
