import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


n_samples = 2
dx = 2
dy = 3
dout = 4
mask_value = -1


class Model(nn.Model):


rnn = nn.LSTM(dy, dout, 1)
# shape (seq_len, batch, input_size)
input = torch.randn(dx, n_samples, dy)
# shape (num_layers * num_directions, batch, hidden_size)
h0 = torch.randn(1, n_samples, dout)
# shape (num_layers * num_directions, batch, hidden_size)
c0 = torch.randn(1, n_samples, dout)
output, (hn, cn) = rnn(input, (h0, c0))
print(input)
print(output)
