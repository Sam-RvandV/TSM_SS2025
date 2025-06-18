import numpy as np
import math
import torch
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from random import randint
from tqdm import trange

class LatentRNN(nn.Module):
    
    def __init__(self, obs_dim, latent_dim, dropout=0.0):
        super(LatentRNN, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.activation = nn.Tanh()

        self.input_to_hidden = nn.Linear(obs_dim, latent_dim, bias=True)
        self.hidden_to_hidden = nn.Linear(latent_dim, latent_dim, bias=False)
        self.hidden_to_output = nn.Linear(latent_dim, obs_dim, bias=True)

        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, time_series, h0):
        seq_len, batch_size, _ = time_series.shape
        h = h0.squeeze(0)  # Convert from (1, B, D) to (B, D)
        outputs = []

        for t in range(seq_len):
            x_t = time_series[t]
            h = self.activation(self.input_to_hidden(x_t) + self.hidden_to_hidden(h))
            h = self.dropout(h)
            x_hat = self.hidden_to_output(h)
            outputs.append(x_hat)

        obs_output = tc.stack(outputs, dim=0)
        return obs_output, h.unsqueeze(0)  # Return hidden state as (1, B, D)

def train(model, data, learning_rate, moment=0, optimizer_function='SGD',
          print_loss=True, batch_size=1, batch_sequence_length=1,
          reg=None, epochs=1000):

    if optimizer_function == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=moment)
    elif optimizer_function == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_function}")

    loss_function = nn.MSELoss()
    losses = []

    obs_dim = model.obs_dim
    latent_dim = model.latent_dim

    if print_loss:
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Optimizer: {optimizer_function}, LR: {learning_rate}, Batch size: {batch_size}, Sequence length: {batch_sequence_length}")

    for epoch in range(epochs):
        h0 = tc.zeros((1, batch_size, latent_dim))

        x = data[:-1]  # Input
        y = data[1:]   # Target

        X = tc.empty((batch_sequence_length, batch_size, obs_dim))
        Y = tc.empty((batch_sequence_length, batch_size, obs_dim))

        for j in range(batch_size):
            ind = tc.randint(0, len(x) - batch_sequence_length + 1, (1,)).item()
            X[:, j, :] = x[ind : ind + batch_sequence_length]
            Y[:, j, :] = y[ind : ind + batch_sequence_length]

        optimizer.zero_grad()
        output, _ = model(X, h0)

        penalty = reg(model.parameters()) if reg is not None else 0.0
        epoch_loss = loss_function(output, Y) + penalty

        epoch_loss.backward()
        optimizer.step()

        losses.append(epoch_loss.item())

        if epoch % 10 == 0 and print_loss:
            print(f"Epoch: {epoch} loss {epoch_loss.item():.6f}")

    return losses
