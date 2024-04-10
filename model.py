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
                nn.Linear(int(hidden_dim*2*3), 1)
            )
        elif parameters["lin_layers"] == 2:
            self.m = nn.Sequential(
                nn.Linear(int(hidden_dim*2*3), int(parameters["hidden1"])),
                nn.ReLU(),
                nn.Linear(int(parameters["hidden1"]), 1)
            )
            
        elif parameters["lin_layers"] == 3:
            self.m = nn.Sequential(
                nn.Linear(int(hidden_dim*2*3), int(parameters["hidden1"])),
                nn.ReLU(),
                nn.Linear(int(parameters["hidden1"]), int(parameters["hidden2"])),
                nn.ReLU(),
                nn.Linear(int(parameters["hidden2"]), 1)
            )      
    
    def forward(self, x):
        y = self.m(x)
        return y
    
class multichannelLSTM(nn.Module):
    
    def __init__(self, LSTM, parameters, hidden_dim):
        super(multichannelLSTM, self).__init__()
        self.LSTM = LSTM
        self.FCN = FCN(parameters, hidden_dim)
        self.flatten = nn.Flatten()
        
    def forward(self, x0, x1, x2):
        x0_out = self.LSTM(x0).unsqueeze(2)
        x1_out = self.LSTM(x1).unsqueeze(2)
        x2_out = self.LSTM(x2).unsqueeze(2)
        
        x_cat = torch.cat((x0_out, x1_out, x2_out), 2)
        
        x_flat = self.flatten(x_cat)

        y = self.FCN(x_flat.float())
        return y
        


