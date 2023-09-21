import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from rgcn_model import RGCN, RGCNLayer
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

    def forward(self, x):
        # pdb.set_trace()
        z_mean, z_log_sigma_sq = self.encoder(x)
        z_sampled = self.sampling(z_mean, z_log_sigma_sq)
        x_output = self.decoder(z_sampled)
        return x_output, z_mean, z_log_sigma_sq

    def vae_loss_function(self, recon_x, x, mu, log_var):
        # pdb.set_trace()
        recon_loss = F.mse_loss(recon_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # disentangle VAE is to add beta before kld_loss
        beta = 1
        loss = recon_loss + beta * kld_loss
        return loss


class GVAE(nn.Module):
    def __init__(self, args, input_dims, interim_size, hidden_dims):
        super(GVAE, self).__init__()
        self.hidden_dims = hidden_dims
        self.rgcn = RGCN(args)
        self.enc_layers = NeuralNetwork(input_dims, [interim_size], hidden_dims)
        self.dec_layers = NeuralNetwork(hidden_dims, [interim_size], input_dims)
        self.enc_weights_mean = nn.Parameter(xavier_uniform_(torch.empty(size=(hidden_dims, hidden_dims))), requires_grad=True)
        self.enc_weights_log_sigma_sq = nn.Parameter(xavier_uniform_(torch.empty(size=(hidden_dims, hidden_dims))), requires_grad=True)
        # self.dec_weights_mean = nn.Parameter(xavier_uniform_(torch.empty(size=(input_dims, input_dims))), requires_grad=True)
        # self.dec_weights_log_sigma_sq = nn.Parameter(xavier_uniform_(torch.empty(size=(input_dims, input_dims))), requires_grad=True)

    def encoder(self, batch):
        # pdb.set_trace()
        z_gcn = self.rgcn(batch)
        z1 = self.enc_layers(z_gcn)
        z_mean = torch.matmul(z1, self.enc_weights_mean)
        z_log_sigma_sq = torch.matmul(z1, self.enc_weights_log_sigma_sq.float())
        return z_gcn, z_mean, z_log_sigma_sq

    def sampling(self, z_mean, z_log_sigma_sq):
        # pdb.set_trace()
        eps = torch.randn(z_mean.size(0), self.hidden_dims).cuda()
        z_sampled = z_mean + torch.sqrt(torch.exp(z_log_sigma_sq)) * eps
        return z_sampled

    def decoder(self, z_sampled):
        out = self.dec_layers(z_sampled)
        return out

    def forward(self, sup_g_bidir):
        # pdb.set_trace()
        x, z_mean, z_log_sigma_sq = self.encoder(sup_g_bidir)
        # pdb.set_trace()
        z_sampled = self.sampling(z_mean, z_log_sigma_sq)
        x_output = self.decoder(z_sampled)

        sup_g_bidir.ndata['g'] = x_output
        return x, x_output, z_mean, z_log_sigma_sq

    def vae_loss_function(self, recon_x, x, mu, log_var):
        # pdb.set_trace()
        recon_loss = F.mse_loss(recon_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # disentangle VAE is to add beta before kld_loss
        beta = 1
        loss = recon_loss + beta * kld_loss
        return loss



class another_GVAE(nn.Module):
    def __init__(self, args, input_dims, hidden_dims):
        super(another_GVAE, self).__init__()
        self.hidden_dims = hidden_dims
        self.enc_layers = RGCN_vae(args, input_dims, [input_dims, input_dims], hidden_dims, is_input_layer=True)
        self.dec_layers = RGCN_vae(args, self.hidden_dims, [input_dims, input_dims], input_dims, activation=None)

        self.enc_weights_mean = nn.Parameter(xavier_uniform_(torch.empty(size=(self.hidden_dims, self.hidden_dims))), requires_grad=True)
        self.enc_weights_log_sigma_sq = nn.Parameter(xavier_uniform_(torch.empty(size=(self.hidden_dims, self.hidden_dims))), requires_grad=True)
        # self.dec_weights_mean = nn.Parameter(xavier_uniform_(torch.empty(size=(input_dims, input_dims))), requires_grad=True)
        # self.dec_weights_log_sigma_sq = nn.Parameter(xavier_uniform_(torch.empty(size=(input_dims, input_dims))), requires_grad=True)

        self.jk_linear = nn.Linear(self.hidden_dims, input_dims)

    def encoder(self, batch):
        # pdb.set_trace()
        z1 = self.enc_layers(batch)
        z_mean = torch.matmul(z1, self.enc_weights_mean)
        z_log_sigma_sq = torch.matmul(z1, self.enc_weights_log_sigma_sq.float())
        return z1, z_mean, z_log_sigma_sq

    def sampling(self, z_mean, z_log_sigma_sq):
        # pdb.set_trace()
        eps = torch.randn(z_mean.size(0), self.hidden_dims).cuda()
        z_sampled = z_mean + torch.sqrt(torch.exp(z_log_sigma_sq)) * eps
        return z_sampled

    def decoder(self, z_sampled):
        out = self.dec_layers(z_sampled)
        return out

    def forward(self, sup_g_bidir):
        # pdb.set_trace()
        x, z_mean, z_log_sigma_sq = self.encoder(sup_g_bidir)
        # pdb.set_trace()
        z_sampled = self.sampling(z_mean, z_log_sigma_sq)

        # sup_g_bidir.ndata['h'] = sup_g_bidir.ndata['h'] + z_sampled
        sup_g_bidir.ndata['h'] = z_sampled
        # is_input_layer=True for decoder if use the 'feat'
        x_output = self.decoder(sup_g_bidir)

        x = self.jk_linear(x)
        sup_g_bidir.ndata['h'] = x
        sup_g_bidir.ndata['g'] = x_output

        return x, x_output, z_mean, z_log_sigma_sq

    def vae_loss_function(self, recon_x, x, mu, log_var):
        # pdb.set_trace()
        recon_loss = F.mse_loss(recon_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # disentangle VAE is to add beta before kld_loss
        beta = 1
        loss = recon_loss + beta * kld_loss
        return loss


class RGCN_vae(nn.Module):
    def __init__(self, args, input_dims, interim_size, output_dims, is_input_layer=False, activation=F.relu):
        super(RGCN_vae, self).__init__()

        self.input_dims = input_dims
        self.interim_size = interim_size
        self.output_dims = output_dims

        self.emb_dim = args.ent_dim
        self.num_rel = args.num_rel
        self.num_bases = args.num_bases
        self.num_layers = args.num_layers
        self.device = args.gpu

        # create rgcn layers
        self.layers = nn.ModuleList()
        self.build_model(is_input_layer, activation)

        # self.jk_linear = nn.Linear(self.emb_dim*(self.num_layers+1), self.emb_dim)

    def build_model(self, is_input_layer=False, activation=F.relu):
        # pdb.set_trace()
        # i2h
        self.layers.append(RGCNLayer(self.input_dims, self.interim_size[0],
                         self.num_rel, self.num_bases, has_bias=True, activation=F.relu,
                         is_input_layer=is_input_layer))
        # h2h
        for idx in range(len(self.interim_size)-1):
            self.layers.append(RGCNLayer(self.interim_size[idx], self.interim_size[idx+1],
                         self.num_rel, self.num_bases, has_bias=True, activation=F.relu))

        self.layers.append(RGCNLayer(self.interim_size[-1], self.output_dims,
                                     self.num_rel, self.num_bases, has_bias=True, activation=activation))

    def forward(self, g):
        for idx, layer in enumerate(self.layers):
            layer(g)

        # pdb.set_trace()
        # g.ndata['h'] = self.jk_linear(g.ndata['repr'])
        # g.ndata['h'] = g.ndata['repr']
        return g.ndata['h']