import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted
from .layers.RevIN import RevIN
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.ndims = args.ndims
        self.use_covar = args.use_covar
        self.norm_mode = args.norm_mode
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, args.d_model, args.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads),
                    args.d_model,
                    2*args.d_model,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        
        # Decoder
        if self.use_covar:
            self.dec_embedding = DataEmbedding_inverted(self.pred_len, args.d_model, args.dropout)
            self.projection = nn.Linear(2*args.d_model, args.pred_len, bias=True)
        else:
            self.projection = nn.Linear(args.d_model, args.pred_len, bias=True)
        
        # Revin
        if self.norm_mode == "revin":
            self.revin = RevIN(self.ndims)


    def forward(self, x_enc, x_mark_enc=None, x_mark_dec=None):
        # Normalization from Non-stationary Transformer
        if self.norm_mode == "norm":
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        elif self.norm_mode == "revin":
            x_enc = self.revin(x_enc, 'norm')
            
        B, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        if self.use_covar:
            future_hint = self.dec_embedding(x_mark_dec.unsqueeze(-1)).expand(-1, N, -1)
            dec_out = torch.cat([enc_out[:,:N,:], future_hint], dim=-1)
            dec_out = self.projection(dec_out).permute(0, 2, 1)[:, :, :N]
        else:
            dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization from Non-stationary Transformer
        if self.norm_mode == "norm":
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        elif self.norm_mode == "revin":
            dec_out = self.revin(dec_out, 'denorm')
        
        return dec_out