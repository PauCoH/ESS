import torch
import torch.nn as nn


class CustomLSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size1=100, hidden_size2=50, mlp_hidden=25, output_size=1, dropout_rate=0.2):
        """
        LSTM model with 2 LSTM layers followed by 2 MLP layers
        
        Args:
            input_size: Number of features in the input (default: 3)
            hidden_size1: Hidden size of the first LSTM layer (default: 100)
            hidden_size2: Hidden size of the second LSTM layer (default: 50)
            mlp_hidden: Hidden size of the first MLP layer (default: 25)
            output_size: Size of the output prediction (default: 1)
            dropout_rate: Dropout probability (default: 0.2)
        """
        super(CustomLSTMModel, self).__init__()
        
                     
        self.lstm1 = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size1, 
                           batch_first=True)
                
        self.lstm2 = nn.LSTM(input_size=hidden_size1, 
                           hidden_size=hidden_size2, 
                           batch_first=True)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        
                    
        self.fc1 = nn.Linear(hidden_size2, mlp_hidden)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(mlp_hidden, output_size)
        
    def forward(self, x, hidden=None):
        """
        Forward pass of the LSTM model
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            hidden: Hidden state tuple for the LSTM (h0, c0) or None
        
        Returns:
            output: Predictions tensor of shape [batch_size, output_size]
            hidden: Tuple of hidden states for the last LSTM layer
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
                                                  
        if hidden is None:
            h1 = torch.zeros(1, batch_size, self.lstm1.hidden_size).to(x.device)
            c1 = torch.zeros(1, batch_size, self.lstm1.hidden_size).to(x.device)
            h2 = torch.zeros(1, batch_size, self.lstm2.hidden_size).to(x.device)
            c2 = torch.zeros(1, batch_size, self.lstm2.hidden_size).to(x.device)
            hidden = ((h1, c1), (h2, c2))
        else:
                                 
            (h1, c1), (h2, c2) = hidden
        
                          
        lstm1_out, (h1, c1) = self.lstm1(x, (h1, c1))
        
                           
        lstm2_out, (h2, c2) = self.lstm2(lstm1_out, (h2, c2))
        lstm2_out = self.dropout1(lstm2_out)
        
                                                    
        last_output = lstm2_out[:, -1, :]
        
                                                                            
        
                    
        fc1_out = self.relu(self.fc1(last_output))
        fc1_out = self.dropout2(fc1_out)
        output = self.fc2(fc1_out)
        
                                                 
        return output, ((h1.detach(), c1.detach()), (h2.detach(), c2.detach()))
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden states
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
        
        Returns:
            tuple: Hidden state tuple ((h1, c1), (h2, c2))
        """
        h1 = torch.zeros(1, batch_size, self.lstm1.hidden_size).to(device)
        c1 = torch.zeros(1, batch_size, self.lstm1.hidden_size).to(device)
        h2 = torch.zeros(1, batch_size, self.lstm2.hidden_size).to(device)
        c2 = torch.zeros(1, batch_size, self.lstm2.hidden_size).to(device)
        
        return ((h1, c1), (h2, c2))

class MLP8(nn.Module):
    def __init__(self, inp_size = 3, batch_norm=False, dropout_var=False, dropout_prob=0.25):
        super(MLP8, self).__init__()
        self.batch_norm = batch_norm
        self.dropout_var = dropout_var
        self.dropout_prob = dropout_prob

        self.fc1 = nn.Linear(inp_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 40)
        self.fc6 = nn.Linear(40, 30)
        self.fc7 = nn.Linear(30, 20)
        self.fc8 = nn.Linear(20, 1)

                                    
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)
            self.bn4 = nn.BatchNorm1d(64)
            self.bn5 = nn.BatchNorm1d(40)
            self.bn6 = nn.BatchNorm1d(30)
            self.bn7 = nn.BatchNorm1d(20)

        self.relu = nn.ReLU()
        if self.dropout_var:
            self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x, hidden=None):
                                                                           
                       
        
                                   
                                                                                                          
        x = self.fc1(x)
                                                    
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc4(x)
                            
                            
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc5(x)
                            
                            
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc6(x)
                            
                            
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc7(x)
                            
                            
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc8(x)

        return x, hidden

class LeakyMLP4(nn.Module):
    def __init__(self, inp_size=3, batch_norm=False, dropout_var=False, dropout_prob=0.25, negative_slope=0.01):
        super(LeakyMLP4, self).__init__()
        self.batch_norm = batch_norm
        self.dropout_var = dropout_var
        self.dropout_prob = dropout_prob

        self.fc1 = nn.Linear(inp_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

                                    
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(64)

                                       
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        
        if self.dropout_var:
            self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x, hidden=None):
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.leaky_relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.leaky_relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.leaky_relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc4(x)

        return x, hidden
