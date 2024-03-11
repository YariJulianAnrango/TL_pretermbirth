import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocessing import train_val_test_split, PrematureDataset

import pandas as pd
from sklearn import metrics

from tqdm import tqdm
import matplotlib.pyplot as plt

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size, layer_dim):
        super(LSTM, self).__init__()
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.target_size = target_size
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.layer_dim, batch_first=True, bidirectional=True, dropout=0.4757946679534617)
        self.lin1 = nn.Linear(2*self.hidden_dim, self.target_size)

    def forward(self, x):

        lstm_out, (h, c) = self.lstm(x.float())
        
        last_layer_hidden_state = h.view(self.layer_dim, 2, x.size(0), self.hidden_dim)[-1]
        
        h_1, h_2 = last_layer_hidden_state[0], last_layer_hidden_state[1]
        final_hidden_state = torch.cat((h_1, h_2), 1)

        logits = self.lin1(final_hidden_state)

        return logits

seed = 1
batch_size = 5

dataset = PrematureDataset("./data/total_df.csv", 1)

train, val, test = train_val_test_split(dataset, 0.3)

train_loader = DataLoader(train, batch_size, shuffle=True)
val_loader = DataLoader(val ,batch_size, shuffle=True)
# dataset_split = TrainValTest(dataset, 0.3, batch_n)
# train, val, test = dataset_split.train, dataset_split.val, dataset_split.test

input_dim = 1
hidden_dim = 6
target_size = 1
layer_dim = 3

model = LSTM(input_dim, hidden_dim, target_size, layer_dim)

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=4.653012126454496*10**-5)

train_loss = []
val_loss_list = []
for epoch in tqdm(range(200)):
    model.train()
    total_loss = 0.0
    
    val_loss = 0
    for sequence, label in train_loader:
        model.zero_grad()
        output = model(sequence)
        label_correct = label.unsqueeze(-1).float()
        loss = loss_function(output, label_correct)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss/len(train)
    train_loss.append(avg_loss)
    
    model.eval()
    for sequence, label in val_loader:
        output = model(sequence)
        label_correct = label.unsqueeze(-1).float()
        vloss = loss_function(output, label_correct)
        
        val_loss += vloss.item()
    avg_val_loss = val_loss/len(val)
    val_loss_list.append(avg_val_loss)

    

plt.plot(train_loss, label = 'train loss')
plt.plot(val_loss_list, label = 'val loss')
plt.title("Results of pca")
plt.legend()
plt.savefig("./figures/pca_results.png")
plt.show()

# Evaluation
model.eval()
preds = []
labels = []

sig = nn.Sigmoid()
for sequence, label in val_loader:
    output = model(sequence)
    pred = sig(output)
    for p in pred:
        preds.append(p.item())
    for l in label:
        labels.append(l.item())
        
fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.title(f"ROC plot with AUC {auc}")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.savefig("./figures/pca_roc.png")
plt.show()


