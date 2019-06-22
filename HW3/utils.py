import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVAE(nn.Module):
    def __init__(self, var_dim):
        super(SimpleVAE, self).__init__()
        self.var_dim = var_dim
        self.fc1 = nn.Linear(2, 32)
        self.enc_mean = nn.Linear(32, 8)
        self.enc_logvar = nn.Linear(32, self.var_dim)
        self.dec_1 = nn.Linear(8, 32)
        self.dec_2 = nn.Linear(32, 2)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.enc_mean(h1), self.enc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.dec_1(z))
        return self.dec_2(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

