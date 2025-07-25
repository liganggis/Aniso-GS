import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from encoding import get_encoder

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x
    

class NLGS(nn.Module):
    def __init__(self, bound: int, tensor_fusion = False):
        super().__init__()
        self.bound = bound
        self.tensor_fusion = tensor_fusion
        # grid
        self.grid_encoder, self.grid_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=4096 * self.bound)

        self.sigma_mlp = MLP(self.grid_in_dim + 8, 1, 64, 2, bias=False)
        # self.register_parameter('alpha', nn.Parameter(torch.zeros(1)))
        
        # view-dependency
        self.view_encoder, self.view_in_dim = get_encoder('sh', input_dim=3, degree=4)
        
        if self.tensor_fusion:
            self.view_mlp = MLP(self.grid_in_dim * self.view_in_dim, 3, 64, 3, bias=False)
        else:
            self.view_mlp = MLP(self.grid_in_dim + self.view_in_dim, 3, 64, 3, bias=False)
        
    def forward(self, x, d, degree):
        h = self.grid_encoder(x, bound=self.bound)
        
        d = self.view_encoder(d)
        if degree < 4:
            ix = degree**2
            # ix = random.randint(0, 15)
            d[..., ix] = 0

        if self.tensor_fusion:
            feat = torch.matmul(h.unsqueeze(-1), d.unsqueeze(1)).view(d.shape[0], -1)
        else:
            feat = torch.cat([h, d], dim=-1)
        color = self.view_mlp(feat)
        color = torch.sigmoid(color)
        
        sigma = self.sigma_mlp(torch.cat([h, d[..., :8]], dim=-1))
        opticity = torch.sigmoid(sigma)
        
        return color, opticity
    
    # optimizer utils
    def grid_params(self):
        return self.grid_encoder.parameters()

    def view_params(self):
        return self.view_mlp.parameters()

    def sigma_params(self):
        return self.sigma_mlp.parameters()

