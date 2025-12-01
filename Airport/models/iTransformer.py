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
        self.norm_mode = args.norm_mode
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(args.seq_len, args.d_model, args.dropout)
        
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
        self.projection = nn.Linear(args.d_model, args.pred_len, bias=True)
        
        # Revin
        if self.norm_mode == "revin":
            self.revin = RevIN(args.ndims)


    def forward(self, x_enc, x_mark_enc=None):
        # Normalization from Non-stationary Transformer
        if self.norm_mode == "norm":
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        elif self.norm_mode == "revin":
            x_enc = self.revin(x_enc, 'norm')
            
        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization from Non-stationary Transformer
        if self.norm_mode == "norm":
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        elif self.norm_mode == "revin":
            dec_out = self.revin(dec_out, 'denorm')

        return dec_out