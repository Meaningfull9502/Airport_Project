import torch
import torch.nn as nn
from .layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.pred_len = args.pred_len
        self.ndims = args.ndims
        self.target = args.target
        self.use_covar = args.use_covar
        
        self.holiday_dim = 16
        self.holiday_embed = nn.Embedding(4, self.holiday_dim)
        
        if self.target == 'both':
            self.lstm = nn.LSTM(
                input_size=self.ndims + self.holiday_dim,
                hidden_size=args.d_model, 
                num_layers=args.n_layers, 
                batch_first=True
            )
        else:
            self.lstm = nn.LSTM(
                input_size=self.ndims, 
                hidden_size=args.d_model, 
                num_layers=args.n_layers, 
                batch_first=True
            )
        
        if self.use_covar:
            self.fc = nn.Linear(args.d_model + (args.pred_len * self.holiday_dim), args.pred_len * self.ndims) 
        else:
            self.fc = nn.Linear(args.d_model, self.pred_len * self.ndims) 
        
        self.norm_mode = args.norm_mode
        if self.norm_mode == "revin":
            self.revin = RevIN(self.ndims)


    def forward(self, x_enc, x_mark_enc=None, x_mark_dec=None):
        batch_size = x_enc.shape[0]
        
        if self.norm_mode == "norm":
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        elif self.norm_mode == "revin":
            x_enc = self.revin(x_enc, 'norm')
            
        if self.target == 'both':
            past_holiday = x_mark_enc.long()
            past_emb = self.holiday_embed(past_holiday) # [B, L, 16]
            x_enc = torch.cat([x_enc, past_emb], dim=-1)
        
        out, _ = self.lstm(x_enc)
        
        if self.use_covar:
            future_holiday = x_mark_dec.long()
            future_emb = self.holiday_embed(future_holiday) # [B, P, 16]
            future_hint = future_emb.reshape(batch_size, -1) # [B, P*16]
            dec_out = torch.cat([out[:,-1], future_hint], dim=-1) # [B, D + P*16]
            dec_out = self.fc(dec_out)
        else:
            dec_out =  self.fc(out[:,-1])
        dec_out = dec_out.view(batch_size, self.pred_len, self.ndims)
                
        if self.norm_mode == "norm":
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        elif self.norm_mode == "revin":
            dec_out = self.revin(dec_out, 'denorm')
        
        return dec_out