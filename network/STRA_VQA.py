import torch
from torch import nn
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import torch.nn.functional as F 


def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# get the mask
def get_attn_pad_mask(video_len, max_len, n_heads, dim):
    """
    :param video_len: [batch_size]
    :param max_len: int
    :param n_heads: int
    :return pad_mask: [batch_size, n_heads, max_len, max_len]
    """
    batch_size = video_len.shape[0]
    global_pad_mask = torch.zeros([batch_size, max_len, max_len])
    local_pad_mask = torch.zeros([batch_size, max_len, dim])
    for i in range(batch_size):
        length = int(video_len[i].item())
        for j in range(length):
            global_pad_mask[i, j, :length] = 1
            local_pad_mask[i, j, :length] = 1

    # pad_mask: [batch_size, n_heads, max_len, max_len]
    global_pad_mask = global_pad_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
    device = video_len.device
    global_pad_mask = global_pad_mask.to(device)
    # local_pad_mask = local_pad_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
    local_pad_mask = local_pad_mask.to(device)
    return [global_pad_mask, local_pad_mask]


def build_net(dim, hidden_dim):
    layers = [
        nn.Linear(2, dim, bias=True), #only frame rate or resolution
        nn.Linear(dim, dim, bias=True),
        nn.Linear(dim, dim * hidden_dim, bias=True),
    ]
    net = nn.Sequential(*layers)
    net.apply(weight_init)
    return net

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x.permute(1, 0, 2) # [batch_size, max_len, d_model] --> [max_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        x = x.permute(1, 0, 2)
        return self.dropout(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, mask, info):
        return self.fn(self.norm(x), mask, info)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.wpn = build_net(dim=dim, hidden_dim=dim)


    def forward(self, x, mask, info):
        b, n,_, = x.shape
        resolution, framerate, _ = info
        resolution = resolution.unsqueeze(1)
        framerate = framerate.unsqueeze(1)
        frate = torch.cat((framerate, resolution), dim=1)
        ff_weight = self.wpn(frate).view(b, self.dim, -1)
        # ff_weight = self.wpn(framerate).view(b, self.dim, -1)
        out = torch.bmm(x, torch.softmax(ff_weight,dim=1))
        return out

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., max_len=600):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.max_len = max_len
        self.attend = nn.Softmax(dim = -1)
        self.wpn_1 = build_net(dim=dim*3, hidden_dim=inner_dim)
        self.wpn_2 = build_net(dim=inner_dim, hidden_dim=dim)
        

    def forward(self, x, mask, info):
        b, n,_, = x.shape
        resolution, framerate, _ = info
        resolution = resolution.unsqueeze(1)
        framerate = framerate.unsqueeze(1)
        frate = torch.cat((framerate, resolution), dim=1)
        qkv_weights = self.wpn_1(frate).view(b, self.dim, -1).chunk(3, dim = -1)
        # qkv_weights = self.wpn_1(framerate).view(b, self.dim, -1).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(torch.bmm(x, torch.softmax(t,dim=1)), 'b n (h d) -> b h n d', h = self.heads), qkv_weights)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = attn * mask[0]

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out_weight = self.wpn_2(frate).view(b,self.inner_dim, -1)
        # out_weight = self.wpn_2(framerate).view(b,self.inner_dim, -1)
        out = torch.bmm(out,  torch.softmax(out_weight,dim=1))
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask, info):
        for attn, ff in self.layers:
            x = attn(x, mask, info) + x
            x = ff(x, mask, info) + x
        return x


def temporal_difference(features):
    b, n, c = features.shape
    mean = features[:,:, :c//2]
    std = features[:,:, c//2:]
    diff_mean = mean[:, 1:,:] - mean[:, :-1,:]
    last_mean = torch.zeros_like(diff_mean[:, -1], device=features.device, dtype=torch.float32)
    diff_mean = torch.cat([last_mean.unsqueeze(1), diff_mean], dim=1)
    diff_std = std[:, 1:,:] + std[:, :-1,:]
    last_std = torch.zeros_like(diff_std[:, -1], device=features.device, dtype=torch.float32)
    diff_std = torch.cat([last_std.unsqueeze(1), diff_std], dim=1)
    return torch.cat([diff_mean, diff_std],dim=2)


def feature_residual(x_dis, x_ref):
    b, n, c = x_dis.shape
    mean = x_ref[:,:, :c//2] - x_dis[:,:, :c//2]
    std = x_ref[:,:, c//2:] + x_dis[:,:, c//2:]
    return torch.cat([mean, std], 2)

class STRA_VQA(nn.Module):
    def __init__(self, *, input_dim=4096, mlp_dim=128, dim_head=64, output_channel=1, depth=5, heads=6, pool = 'reg', dropout = 0.1, emb_dropout = 0.1, max_length=600):
        super().__init__()
        
        assert pool in {'reg', 'mean'}, 'pool type must be either reg (reg token) or mean (mean pooling)'
        #reduce the dimension of the input embeddings
        self.reduce_embedding = nn.Linear(input_dim*2, mlp_dim, bias=True)
        # self.embedding = Embedding(input_dim, mlp_dim, step=20, max_len=max_length)
        self.max_length = max_length
        self.pos_embedding = PositionalEncoding(mlp_dim, max_len=max_length+1)
        # self.pos_embedding = nn.Parameter(torch.randn(1, max_length + 1, mlp_dim))
        self.reg_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
        # self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(mlp_dim, depth, heads, dim_head, mlp_dim*4, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, output_channel, bias=True)
        )
        self.heads = heads
        self.mlp_dim = mlp_dim
        weight_init(self.reduce_embedding)
        weight_init(self.mlp_head)


    def forward(self, input, info, video_len):
        x_ref, x_dis = input
        x_diff = feature_residual(x_dis, x_ref)
        x = torch.cat([x_dis, x_diff], dim=2)
        x = self.reduce_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.reg_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_embedding(x)

        pad_mask = get_attn_pad_mask(video_len+1, self.max_length+1, self.heads, self.mlp_dim)
        x = x*pad_mask[1]
        x = self.transformer(x, pad_mask, info)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
