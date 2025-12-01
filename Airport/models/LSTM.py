import torch
import torch.nn as nn
from .layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.pred_len = args.pred_len
        self.ndims = args.ndims
        
        self.lstm = nn.LSTM(
            input_size=self.ndims, 
            hidden_size=args.d_model, 
            num_layers=args.n_layers, 
            batch_first=True
        )
        
        self.fc = nn.Linear(args.d_model, self.pred_len * self.ndims) 
        
        self.norm_mode = args.norm_mode
        if self.norm_mode == "revin":
            self.revin = RevIN(self.ndims)


    def forward(self, x_enc):
        batch_size = x_enc.shape[0]
        
        if self.norm_mode == "norm":
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        elif self.norm_mode == "revin":
            x_enc = self.revin(x_enc, 'norm')

        out, _ = self.lstm(x_enc)
        dec_out =  self.fc(out[:,-1,:])
        dec_out = dec_out.view(batch_size, self.pred_len, self.ndims)
                
        if self.norm_mode == "norm":
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        elif self.norm_mode == "revin":
            dec_out = self.revin(dec_out, 'denorm')
        
        return dec_out