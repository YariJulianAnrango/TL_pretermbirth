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

    def __init__(self, parameters):
        super(LSTM, self).__init__()
        self.layer_dim = parameters["layer_dim"]
        self.input_dim = 3
        self.hidden_dim = parameters["hidden_dim"]
        self.target_size = 1
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.layer_dim, batch_first=True, bidirectional=True, dropout=parameters["dropout"])
        self.lin1 = nn.Linear(2*self.hidden_dim, self.target_size)

    def forward(self, x):

        lstm_out, (h, c) = self.lstm(x.float().to("mps"))
        
        h0, c0 = self.init_hidden(x)
        lstm_out, (h, c) = self.lstm(x.float(), (h0, c0))
        last_layer_hidden_state = h.view(self.layer_dim, 2, x.size(0), self.hidden_dim)[-1]
        
        h_1, h_2 = last_layer_hidden_state[0], last_layer_hidden_state[1]
        final_hidden_state = torch.cat((h_1, h_2), 1)

        logits = self.lin1(final_hidden_state)

        return logits


def train_and_evaluate(parameters, model):
    
    dataset = PrematureDataset("./data/total_df.csv")
    
    def init_hidden(self, x):
        h0 = torch.zeros(2*self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(2*self.layer_dim, x.size(0), self.hidden_dim)
        return h0, c0

# seed = 1
# batch_size = 5

# dataset = PrematureDataset("./data/total_df.csv", 1)

    train, val, test = train_val_test_split(dataset, 0.3)
    
    train_loader = DataLoader(train, parameters["batch_size"], shuffle=True)
    val_loader = DataLoader(val ,parameters["batch_size"], shuffle=True)
    
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else "cpu")
    print(use_mps)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = getattr(optim, parameters['optimizer'])(model.parameters(), lr= parameters['learning_rate'])

    if use_mps:
        model = model.to(device)
        loss_function = loss_function.to(device)
        
    train_loss = []
    val_loss_list = []
    for epoch in tqdm(range(50)):
        model.train()
        total_loss = 0.0
    
#         val_loss = 0
#         for sequence, label in train_loader:    
#             model.zero_grad()
#             
            sequence.float().to(device)
            label.to(device)
            
            output = model(sequence)
#             label_correct = label.unsqueeze(-1).float().to(device)
#             loss = loss_function(output, label_correct)
#             
            loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#         avg_loss = total_loss/len(train)
#         train_loss.append(avg_loss)
    
#         model.eval()
#         for sequence, label in val_loader:
#             sequence.float().to(device)
            
            output = model(sequence)
#             label_correct = label.unsqueeze(-1).float().to(device)
#             vloss = loss_function(output, label_correct)
        
            val_loss += vloss.item()
        avg_val_loss = val_loss/len(val)
        val_loss_list.append(avg_val_loss)
        

    model.eval()
    preds = []
    labels = []
#         val_loss += vloss.item()
#     avg_val_loss = val_loss/len(val)
#     val_loss_list.append(avg_val_loss)

    

# plt.plot(train_loss, label = 'train loss')
# plt.plot(val_loss_list, label = 'val loss')
# plt.title("Results of pca")
# plt.legend()
# plt.savefig("./figures/pca_results.png")
# plt.show()

# # Evaluation
# model.eval()
# preds = []
# labels = []

#     sig = nn.Sigmoid()
#     for sequence, label in val_loader:
#         sequence.float().to(device)
        
        output = model(sequence)
#         pred = sig(output)
#         for p in pred:
#             preds.append(p.item())
#         for l in label:
#             labels.append(l.item())
        
fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
print(metrics.auc(fpr, tpr))

plt.plot(fpr, tpr)
plt.show()


