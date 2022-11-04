import torch
import torch.nn as nn
from util import DataMode

class WindowMLP(nn.Module):
    def __init__(self, device, num_features, hidden_size, num_layers, dropout, window_size, data_mode=None, seq_length=None):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.seq_length = seq_length
        self.window_size = window_size
        self.output_size = 2 if data_mode == DataMode.BOTH else 1

        self.in_size = self.num_features*self.window_size

        self.lin1 = nn.Linear(self.in_size, self.in_size//8)
        self.lin2 = nn.Linear(self.in_size//8, self.in_size//16)
        self.lin3 = nn.Linear(self.in_size//16, self.in_size//32)
        self.lin4 = nn.Linear(self.in_size//32, self.output_size)

        self.drop = nn.Dropout(p=dropout)
        self.act = nn.Tanh()

    def forward(self, x):

        output_total = torch.zeros((x.shape[0], x.shape[1], self.output_size)).to(self.device)
        input_tensor = torch.zeros((x.shape[0], x.shape[1]+self.window_size, x.shape[2])).to(self.device)

        input_tensor[:, :self.window_size] = x[:, 0].unsqueeze(1)
        input_tensor[:, self.window_size:] = x

        last_row = torch.zeros(x.shape[0], self.output_size).to(self.device)

        # Window MLP, so loop through the data based on the window 
        for i in range(x.shape[1]):
            # Get current 'window' of data
            x_cropped = input_tensor[:,i:i+self.window_size].view(-1, self.in_size).to(self.device)

            # Fwd pass through MLP
            x_cropped = self.drop(self.act(self.lin1(x_cropped)))
            x_cropped = self.drop(self.act(self.lin2(x_cropped)))
            x_cropped = self.drop(self.act(self.lin3(x_cropped)))
            out = self.drop(self.act(self.lin4(x_cropped)))

            # If the output predicted is 0, just take the previous value
            # Weird thing that seemed to happen with the window MLP is that
            # it will juts predict 0 randomly sometimes
            # This works fine as a fix and was fast to implement
            mask = torch.ne(out, torch.zeros(out.shape).to(self.device))
            output_total[:,i] = last_row
            output_total[:,i][mask] = out[mask]

            last_row = output_total[:,i]

        return output_total

class RegressionLSTM(nn.Module):
    def __init__(self, device, num_features, hidden_size, num_layers, dropout, window_size, data_mode=None, seq_length=None):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # If both need 2 outputs for both position and velocity, else only need 1 output
        self.output_size = 2 if data_mode == DataMode.BOTH else 1
        print('Output size = %d' %(self.output_size))

        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=dropout
        )

        self.output_linear = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size)
        self.output_linear_final = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size)

    def init_model_state(self, batch_size):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).requires_grad_().to(self.device)
        return (h0, c0)

    def forward(self, x):
        # Straightforward fwd pass
        batch_size = x.shape[0]

        h0, c0 = self.init_model_state(batch_size)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        out = self.output_linear(out)
        out = self.output_linear_final(out)

        return out

