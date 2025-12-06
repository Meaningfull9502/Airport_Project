import torch
from torch import nn
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import PatchEmbedding, Patching
from .layers.RevIN import RevIN


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """
    def __init__(self, args):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        args.stride = args.patch_len  # non-overlapping patching
        self.ndims = args.ndims
        self.use_covar = args.use_covar
        self.padding = 0
        
        # patching and embedding
        if self.use_covar:
            self.patch_embedding = PatchEmbedding(args.d_model, 2*args.patch_len, 2*args.stride, self.padding, args.dropout)
            self.dec_embedding = nn.Linear(args.pred_len, args.d_model)
        else:
            self.patch_embedding = PatchEmbedding(args.d_model, args.patch_len, args.stride, self.padding, args.dropout)
            
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
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(args.d_model), Transpose(1,2))
        )

        # Prediction Head
        if self.use_covar:
            self.head_nf = args.d_model * int((args.seq_len / args.patch_len) + 1)
        else:
            self.head_nf = args.d_model * int(args.seq_len / args.patch_len)
        self.head = FlattenHead(self.ndims, self.head_nf, args.pred_len, head_dropout=args.dropout)

        self.norm_mode = args.norm_mode
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
        
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = torch.cat([x_enc, x_mark_enc.unsqueeze(-1).permute(0, 2, 1)], dim=-1) if self.use_covar else x_enc
        enc_out, n_vars = self.patch_embedding(x_enc)  # u: [bs * nvars x patch_num x d_model]
            
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, _ = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        # Decoder
        if self.use_covar:
            dec_out = self.dec_embedding(x_mark_dec.unsqueeze(-1).permute(0, 2, 1))
            dec_out = dec_out.unsqueeze(-1)  # z: [bs x nvars x pred_len x d_model]
            enc_out = torch.cat([enc_out, dec_out], dim=-1)
        
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        if self.norm_mode == "norm":
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        elif self.norm_mode == "revin":
            dec_out = self.revin(dec_out, 'denorm')
            
        return dec_out