import torch
import torch as t
import torch.nn as nn
import numpy as np
from typing import Tuple
from .layers.RevIN import RevIN


class NBeatsBlock(t.nn.Module):
    def __init__(self, input_size, theta_size: int, basis_function: t.nn.Module, layers: int, layer_size: int):
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [t.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = t.nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = t.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)

class NBeats(t.nn.Module):
    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast

class GenericBasis(t.nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]

class Model(t.nn.Module):
    """
    N-BEATS Base Model
    """
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = args.ndims
        
        # 단변량만 지원
        if self.enc_in != 1:
             raise ValueError("N-BEATS는 현재 enc_in=1 (단변량)만 지원합니다.")

        backcast_size = self.seq_len
        forecast_size = self.pred_len
        layer_size = args.d_model
        layers = args.n_layers
        num_blocks = args.n_layers

        blocks = t.nn.ModuleList()
        theta_size = backcast_size + forecast_size

        for _ in range(num_blocks):
            basis = GenericBasis(backcast_size=backcast_size, forecast_size=forecast_size)
            block = NBeatsBlock(
                input_size=backcast_size,
                theta_size=theta_size,
                basis_function=basis,
                layers=layers,
                layer_size=layer_size,
            )
            blocks.append(block)

        self.nbeats = NBeats(blocks)
        
        self.norm_mode = args.norm_mode
        if self.norm_mode == "revin":
            self.revin = RevIN(args.ndims)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, **kwargs):
        if self.norm_mode == "norm":
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        elif self.norm_mode == "revin":
            x_enc = self.revin(x_enc, 'norm')

        # [B, L, 1] -> [B, L]
        x = x_enc.squeeze(-1)
        
        input_mask = t.ones_like(x)
        forecast = self.nbeats(x, input_mask) # [B, Pred]

        # [B, Pred] -> [B, Pred, 1]
        dec_out = forecast.unsqueeze(-1)
        
        if self.norm_mode == "norm":
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        elif self.norm_mode == "revin":
            dec_out = self.revin(dec_out, 'denorm')
        
        return dec_out