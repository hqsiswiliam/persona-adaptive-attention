import torch
from torch import nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size,
                 layers, dropout, bidirectional, rnn_class):
        super(DecoderRNN, self).__init__()
        if bidirectional:
            hidden_size *= 2
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
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = RNN_CLASS(hidden_size, hidden_size,
                             batch_first=True, num_layers=layers,
                             dropout=dropout, bidirectional=False)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.layers = layers
        self.rnn_class = rnn_class

    def expand_hidden(self, hidden):
        if self.rnn_class == 'lstm':
            # only fetch the last layer
            h_hidden, c_hidden = hidden
            h_hidden = h_hidden[-1:]
            c_hidden = c_hidden[-1:]
            # expand the hidden state to each rnn layer, [1, L, D] -> [Layers, L, D]
            # expand the cell state to each rnn layer, [1, L, D] -> [Layers, L, D]
            hidden_state = h_hidden.repeat(self.layers, 1,1)
            cell_state = c_hidden.repeat(self.layers, 1,1)
            return hidden_state, cell_state
        if self.rnn_class == 'gru' or self.rnn_class == 'rnn':
            # only fetch the last layer
            h_hidden = hidden[-1:]
            # expand the hidden state to each rnn layer, [1, L, D] -> [Layers, L, D]
            hidden_state = h_hidden .repeat(self.layers, 1,1)
            return hidden_state
        raise NotImplementedError("Need implementation")

    def forward(self, input, hidden):
        expanded_hidden = self.expand_hidden(hidden)
        embedding_output = self.embedding(input)
        relu_output = F.relu(embedding_output)
        rnn_output, rnn_hidden = self.rnn(relu_output, expanded_hidden)
        linear_output = self.out(rnn_output)
        softmax_output = self.softmax(linear_output)
        return softmax_output, rnn_hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)
#
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))
#
#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#
#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
