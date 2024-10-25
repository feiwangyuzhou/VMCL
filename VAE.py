import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import pdb

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, interim_size, output_size):
        super(NeuralNetwork, self).__init__()
        if type(interim_size) != list:
            raise ValueError('Interim_size should be a non-empty list.')
        modules = list()
        self.nn_layers = list()
        if not interim_size:
            self.nn_layers.append(nn.Linear(input_size, output_size))
        else:
            self.nn_layers.append(nn.Linear(input_size, interim_size[0]))
            for i in range(len(interim_size)-1):
                self.nn_layers.append(nn.Linear(interim_size[i], interim_size[i+1]))
            self.nn_layers.append(nn.Linear(interim_size[-1], output_size))
        for l in self.nn_layers:
            modules.extend([l, nn.Tanh()])
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    def regularizer(self, p):
        return sum([torch.norm(x.weight, p=p) for x in self.nn_layers])

class VAE(nn.Module):
    def __init__(self, input_dims, interim_size, hidden_dims):
        super(VAE, self).__init__()
        self.hidden_dims = hidden_dims
        self.enc_layers = NeuralNetwork(input_dims, [interim_size], hidden_dims)
        self.dec_layers = NeuralNetwork(hidden_dims, [interim_size], input_dims)
        self.enc_weights_mean = nn.Parameter(xavier_uniform_(torch.empty(size=(hidden_dims, hidden_dims))), requires_grad=True)
        self.enc_weights_log_sigma_sq = nn.Parameter(xavier_uniform_(torch.empty(size=(hidden_dims, hidden_dims))), requires_grad=True)
        # self.dec_weights_mean = nn.Parameter(xavier_uniform_(torch.empty(size=(input_dims, input_dims))), requires_grad=True)
        # self.dec_weights_log_sigma_sq = nn.Parameter(xavier_uniform_(torch.empty(size=(input_dims, input_dims))), requires_grad=True)
        self.dec_L = 1
        self.fusion='add'
        if self.fusion == 'cat':
            self.W = nn.Parameter(xavier_uniform_(torch.empty(size=(input_dims*self.dec_L, input_dims))), requires_grad=True)

    def encoder(self, batch):
        z1 = self.enc_layers(batch)
        z_mean = torch.matmul(z1, self.enc_weights_mean)
        z_log_sigma_sq = torch.matmul(z1, self.enc_weights_log_sigma_sq.float())
        return z_mean, z_log_sigma_sq

    def sampling(self, z_mean, z_log_sigma_sq):
        # pdb.set_trace()
        eps = torch.randn(z_mean.size(0), self.hidden_dims).cuda()
        z_sampled = z_mean + torch.sqrt(torch.exp(z_log_sigma_sq)) * eps
        return z_sampled

    def decoder(self, z_sampled):
        out = self.dec_layers(z_sampled)
        return out

    # add
    def myAdd(self, z_mean, z_log_sigma_sq):
        # pdb.set_trace()
        x_output = 0
        for x_i in range(self.dec_L):
            z_sampled = self.sampling(z_mean, z_log_sigma_sq)
            x_output += self.decoder(z_sampled)
        return x_output

    # cat
    def myCat(self, z_mean, z_log_sigma_sq):
        x_output_list = []
        for x_i in range(self.dec_L):
            z_sampled = self.sampling(z_mean, z_log_sigma_sq)
            x_output_list.append(self.decoder(z_sampled))
        # pdb.set_trace()
        x_output = torch.cat(x_output_list, 1)
        x_output = torch.mm(x_output, self.W)
        return x_output

    def forward(self, x):
        # pdb.set_trace()
        z_mean, z_log_sigma_sq = self.encoder(x)
        # pdb.set_trace()
        if self.fusion == 'add':
            x_output = self.myAdd(z_mean, z_log_sigma_sq)
        elif self.fusion == 'cat':
            x_output = self.myCat(z_mean, z_log_sigma_sq)
        else:
            z_sampled = self.sampling(z_mean, z_log_sigma_sq)
            x_output = self.decoder(z_sampled)

        return x_output, z_mean, z_log_sigma_sq



    def vae_loss_function(self, recon_x, x, mu, log_var):
        # pdb.set_trace()
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        recon_loss = F.mse_loss(recon_x, x)

        # disentangle VAE is to add beta before kld_loss
        beta = 1
        loss = recon_loss + beta * kld_loss
        return loss