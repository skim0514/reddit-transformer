import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SummarizationRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        #add Models

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = 1

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size, self.layers)
        self.gru2 = nn.GRU(hidden_size, hidden_size, self.layers)
        self.gru3 = nn.GRU(hidden_size, hidden_size, self.layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        encoded = self.encoder(input).view(1,1,-1)
        output, hidden = self.gru1(encoded, hidden)
        output, hidden = self.gru2(output, hidden)
        output, hidden = self.gru3(output, hidden)
        output = self.decoder(output)
        return output[0], hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)




