import torch
import torch.nn as nn
import sys
sys.path.append('../')
from .layers.RevIN import RevIN


class ResBlock(nn.Module):
    def __init__(self, args):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(args.seq_len, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, args.seq_len),
            nn.Dropout(args.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(args.ndims, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, args.ndims),
            nn.Dropout(args.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x
    
    
class TempBlock(nn.Module):
    def __init__(self, args):
        super(TempBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(args.seq_len, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, args.seq_len),
            nn.Dropout(args.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)

        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.ndims == 1:
            self.model = nn.ModuleList([TempBlock(args) for _ in range(args.n_layers)])
        else:
            self.model = nn.ModuleList([ResBlock(args) for _ in range(args.n_layers)])
        self.pred_len = args.pred_len
        self.layers = args.n_layers
        self.projection = nn.Linear(args.seq_len, args.pred_len)
        self.norm_mode = args.norm_mode
        if self.norm_mode == "revin":
            self.revin = RevIN(args.ndims)

    def forward(self, x_enc):
        if self.norm_mode == "norm":
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        elif self.norm_mode == "revin":
            x_enc = self.revin(x_enc, 'norm')

        for i in range(self.layers):
            x_enc = self.model[i](x_enc)
        dec_out = self.projection(x_enc.transpose(1,2)).transpose(1,2)
        
        if self.norm_mode == "norm":
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        elif self.norm_mode == "revin":
            dec_out = self.revin(dec_out, 'denorm')
        
        return dec_out