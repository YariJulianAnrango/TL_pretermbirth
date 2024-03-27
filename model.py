import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, parameters, target_size = 1, input_dim = 3):
        super(LSTM, self).__init__()
        self.layer_dim = parameters["layer_dim"]
        self.input_dim = input_dim
        self.hidden_dim = parameters["hidden_dim"]
        self.target_size = target_size
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.layer_dim, batch_first=True, bidirectional=True, dropout=parameters["dropout"])
        self.lin1 = nn.Linear(2*self.hidden_dim, self.target_size)

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        lstm_out, (h, c) = self.lstm(x.float(), (h0, c0))
        
        last_layer_hidden_state = h.view(self.layer_dim, 2, x.size(0), self.hidden_dim)[-1]
        
        h_1, h_2 = last_layer_hidden_state[0], last_layer_hidden_state[1]
        final_hidden_state = torch.cat((h_1, h_2), 1)

        logits = self.lin1(final_hidden_state)

        return logits
    
    def init_hidden(self, x):
        h0 = torch.zeros(2*self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(2*self.layer_dim, x.size(0), self.hidden_dim)
        return h0, c0




