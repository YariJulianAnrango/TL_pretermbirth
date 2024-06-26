import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, parameters, target_size = 1, input_dim = 3, device = "cpu"):
        super(LSTM, self).__init__()
        self.layer_dim = int(parameters["layer_dim"])
        self.input_dim = input_dim
        self.hidden_dim = int(parameters["hidden_dim"])
        self.target_size = target_size
        self.device = device
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.layer_dim, batch_first=True, bidirectional=True, dropout=parameters["dropout"])
        self.lin1 = nn.Linear(2*self.hidden_dim, self.target_size)

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        lstm_out, (h, c) = self.lstm(x.float(), (h0.to(self.device), c0.to(self.device)))
        
        last_layer_hidden_state = h.view(self.layer_dim, 2, x.size(0), self.hidden_dim)[-1]
        
        h_1, h_2 = last_layer_hidden_state[0], last_layer_hidden_state[1]
        final_hidden_state = torch.cat((h_1, h_2), 1)
        logits = self.lin1(final_hidden_state)

        return logits
    
    def init_hidden(self, x):
        h0 = torch.zeros(2*self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(2*self.layer_dim, x.size(0), self.hidden_dim)
        return h0, c0

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class FCN(nn.Module):
    def __init__(self, parameters, hidden_dim):
        super(FCN, self).__init__()
        if parameters["lin_layers"] == 1:
            self.m = nn.Sequential(
                nn.Linear(int(hidden_dim*3), 1)
            )
        elif parameters["lin_layers"] == 2:
            self.m = nn.Sequential(
                nn.Linear(int(hidden_dim*3), int(parameters["hidden1"])),
                nn.ReLU(),
                nn.Linear(int(parameters["hidden1"]), 1)
            )
            
        elif parameters["lin_layers"] == 3:
            self.m = nn.Sequential(
                nn.Linear(int(hidden_dim*3), int(parameters["hidden1"])),
                nn.ReLU(),
                nn.Linear(int(parameters["hidden1"]), int(parameters["hidden2"])),
                nn.ReLU(),
                nn.Linear(int(parameters["hidden2"]), 1)
            )      
    
    def forward(self, x):
        y = self.m(x)
        return y
    
class MultiChannelModel(nn.Module):
    
    def __init__(self, model, parameters, hidden_dim):
        super(MultiChannelModel, self).__init__()
        self.model = model
        self.FCN = FCN(parameters, hidden_dim)
        self.flatten = nn.Flatten()
        self.batch = nn.BatchNorm1d(3)
        
    def forward(self, x):
        self.batch(x)
        
        x0, x1, x2 = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        
        x0_out = self.model(x0.unsqueeze(1))
        x1_out = self.model(x1.unsqueeze(1))
        x2_out = self.model(x2.unsqueeze(1))
        
        x_cat = torch.cat((x0_out.unsqueeze(2), x1_out.unsqueeze(2), x2_out.unsqueeze(2)), 2)
   
        x_flat = self.flatten(x_cat)
        y = self.FCN(x_flat.float())
        return y

class CNN(nn.Module):
    def __init__(self, target_size, input_dim = 1):
        super(CNN, self).__init__()        
        self.conv1 = nn.Conv1d(input_dim, 128, 8)
        self.batch1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(128, 256, 5)
        self.batch2 = nn.BatchNorm1d(256)
        
        
        self.conv3 = nn.Conv1d(256, 128, 3)
        self.batch3 = nn.BatchNorm1d(128)
        
        self.lin1 = nn.Linear(128, target_size)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x1 = self.relu(self.batch1(self.conv1(x)))
        x2 = self.relu(self.batch2(self.conv2(x1)))
        x3 = self.relu(self.batch3(self.conv3(x2)))

        x4 = x3.mean(2)

        y = self.lin1(x4)
        
        return y
        


