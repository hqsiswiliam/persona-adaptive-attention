import torch
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,
                 layers, dropout, bidirectional, rnn_class):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        RNN_CLASS = nn.GRU
        if rnn_class == 'lstm':
            RNN_CLASS = nn.LSTM
        elif rnn_class == 'rnn':
            RNN_CLASS = nn.RNN
        elif rnn_class == 'gru':
            RNN_CLASS = nn.GRU
        else:
            raise ValueError("No rnn class specified")
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.bidirectional = bidirectional
        self.rnn = RNN_CLASS(hidden_size, hidden_size,
                             batch_first=True, num_layers=layers,
                             dropout=dropout, bidirectional=bidirectional)
        self.rnn_class = rnn_class

    def forward(self, input):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.rnn(output)
        if self.bidirectional and self.rnn_class == 'lstm':
            hidden = (torch.cat((hidden[0][0], hidden[0][1]), dim=-1).unsqueeze(0),
                      torch.cat((hidden[1][0], hidden[1][1]), dim=-1).unsqueeze(0))
        elif self.bidirectional:
            hidden = torch.cat((hidden[0], hidden[1]), dim=-1).unsqueeze(0)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
