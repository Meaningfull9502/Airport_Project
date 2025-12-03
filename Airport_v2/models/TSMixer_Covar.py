import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from .layers.RevIN import RevIN


class FeatureMixing(nn.Module):
    """
    Feature Mixing Layer (MLP on channel dimension)
    - 입력 채널과 출력 채널이 다르면 Projection을 통해 차원을 맞춤
    """
    def __init__(self, args, in_channels, out_channels):
        super(FeatureMixing, self).__init__()
        
        self.norm_before = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, args.d_model)
        self.fc2 = nn.Linear(args.d_model, out_channels)
        self.dropout = nn.Dropout(args.dropout)
        
        # Residual Connection을 위한 Projection
        if in_channels != out_channels:
            self.projection = nn.Linear(in_channels, out_channels)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        # x: [Batch, Length, In_Channels]
        x_proj = self.projection(x)
        
        x = self.norm_before(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x_proj + x


class MixerBlock(nn.Module):
    """
    TSMixer Block: Time Mixing -> Feature Mixing
    """
    def __init__(self, args, in_channels, out_channels):
        super(MixerBlock, self).__init__()
        
        # 1. Time Mixing (시간 축 섞기)
        self.norm_time = nn.LayerNorm(in_channels)
        # 이미 L -> P로 변환된 상태에서 들어오므로 길이는 pred_len
        self.time_linear = nn.Linear(args.pred_len, args.pred_len)
        self.dropout = nn.Dropout(args.dropout)

        # 2. Feature Mixing (채널 축 섞기)
        self.feature_mixing = FeatureMixing(args, in_channels, out_channels)

    def forward(self, x):
        # x: [Batch, Length, Channels]
        # --- Time Mixing ---
        res = x
        x = self.norm_time(x)
        x = x.transpose(1, 2)      # [B, C, L]
        x = self.time_linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)      # [B, L, C]
        x = x + res                # Residual

        # --- Feature Mixing ---
        x = self.feature_mixing(x)
        
        return x


class Model(nn.Module):
    """
    TSMixerExt Unified Model
    Args (in args object):
        seq_len, pred_len, 
        enc_in (input channels), c_out (output channels),
        c_extra (covariate channels - set to 0 if unused),
        d_model, n_layers, dropout, norm_mode ('revin' or 'norm')
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.norm_mode = args.norm_mode
        self.ndims = args.ndims
        self.use_covariates = args.use_covar

        # 1. Input Transformations (Time Projection)
        # 과거 데이터의 길이를 미리 미래 예측 길이로 변환
        self.fc_hist = nn.Linear(args.seq_len, args.pred_len)
        self.fc_out = nn.Linear(args.d_model, self.ndims)

        # 2. Embedding Layers (Feature Mixing)
        # (A) History Branch
        # 공변량 사용 시: 입력 채널 = ndims + 1
        # 미사용 시: 입력 채널 = enc_in
        hist_in_channels = self.ndims + 1
        self.feature_mixing_hist = FeatureMixing(
            args, 
            in_channels=hist_in_channels, 
            out_channels=args.d_model
        )
        
        # (B) Future Branch
        if self.use_covariates:
            self.feature_mixing_future = FeatureMixing(
                args, 
                in_channels=1, 
                out_channels=args.d_model
            )

        # 3. Mixer Blocks Setup
        # 첫 번째 블록의 입력 채널 결정
        # 공변량 사용 시: History(d_model) + Future(d_model) = 2 * d_model
        # 미사용 시: History(d_model) = d_model
        first_in_dim = 2 * args.d_model if self.use_covariates else args.d_model
        
        # 채널 리스트 생성
        # 예: n_layers=3, d_model=64, use_cov=True -> [128, 64, 64, 64]
        # zip 결과 -> (128,64), (64,64), (64,64) -> 총 3개 레이어 생성됨
        channels = [first_in_dim] + [args.d_model] * (args.n_layers-1)
        
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(args, in_ch, out_ch)
            for in_ch, out_ch in zip(channels[:-1], channels[1:])
        ])

        # Normalization Init
        if self.norm_mode == 'revin':
            self.revin = RevIN(self.ndims)

    def forward(self, x_hist, x_extra_hist=None, x_extra_future=None):
        # x_hist: [Batch, seq_len, enc_in]
        # x_extra_hist: [Batch, seq_len, 1] (Optional)
        # x_extra_future: [Batch, pred_len, 1] (Optional)

        # --- 1. Normalization ---
        if self.norm_mode == "norm":
            means = x_hist.mean(1, keepdim=True).detach()
            x_hist = x_hist - means
            stdev = torch.sqrt(torch.var(x_hist, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_hist /= stdev
        elif self.norm_mode == "revin":
            x_hist = self.revin(x_hist, 'norm')

        # --- 2. Prepare History ---
        # 공변량이 있으면 과거 데이터에 결합
        x_hist_combined = torch.cat([x_hist, x_extra_hist.unsqueeze(-1)], dim=-1)
        
        # Time Projection: 길이 L -> 길이 P
        # (B, L, C) -> (B, C, L) -> Linear -> (B, C, P) -> (B, P, C)
        x_hist_temp = x_hist_combined.transpose(1, 2)
        x_hist_temp = self.fc_hist(x_hist_temp)
        x_hist_combined = x_hist_temp.transpose(1, 2)
        
        # Feature Mixing (Embedding)
        x_hist_emb = self.feature_mixing_hist(x_hist_combined)

        # --- 3. Prepare Future & Combine ---
        if self.use_covariates:
            x_future_emb = self.feature_mixing_future(x_extra_future.unsqueeze(-1))
            x = torch.cat([x_hist_emb, x_future_emb], dim=-1)  # Concat: [B, P, d_model] + [B, P, d_model] -> [B, P, 2*d_model]
        else:
            x = x_hist_emb

        # --- 4. Mixer Blocks ---
        # 첫 블록에서 차원이 2*d_model -> d_model로 압축됨 (공변량 모드 시)
        for block in self.mixer_blocks:
            x = block(x)

        # --- 5. Final Output Projection ---
        dec_out = self.fc_out(x)

        # --- 6. De-Normalization ---
        if self.norm_mode == "norm":
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        elif self.norm_mode == "revin":
            dec_out = self.revin(dec_out, 'denorm')

        return dec_out