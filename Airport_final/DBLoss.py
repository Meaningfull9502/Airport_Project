import torch
import torch.nn as nn

class DBLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1e-8):
        """
        DBLoss: Decomposition-based Loss Function
        
        Args:
            alpha (float): EMA 스무딩 팩터 (0 < alpha < 1). 
                           trend와 seasonal 분해를 조절합니다. 
            beta (float): 가중치 파라미터 (0 <= beta <= 1).
                          Seasonal Loss와 Trend Loss의 비중을 조절합니다.
            epsilon (float): 0으로 나누는 것을 방지하기 위한 작은 상수.
        """
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss() # L_S는 L2 norm 기반
        self.l1_loss = nn.L1Loss()   # L_T는 L1 norm 기반

    def ema_decomposition(self, x):
        """
        Algorithm 1: Calculation Process of EMA Decomposition Module 
        입력 시계열을 Trend와 Seasonal 성분으로 분해합니다.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Channels)
            
        Returns:
            seasonal (torch.Tensor): Seasonal component
            trend (torch.Tensor): Trend component
        """
        B, T, N = x.shape
        device = x.device
        
        # 1. 가중치 W 계산: [(1-alpha)^(T-1), ..., (1-alpha)^0]
        arange = torch.arange(T, device=device).float()
        weights = (1.0 - self.alpha) ** (T - 1 - arange)
        
        # Broadcasting을 위해 차원 맞춤 (1, T, 1) 
        weights = weights.view(1, T, 1)
        
        # 2. Weighted Cumulative Sum 계산 (Vectorized EMA) 
        # Trend = cumsum(X * W) / cumsum(W)
        weighted_x = x * weights
        cum_weighted_x = torch.cumsum(weighted_x, dim=1)
        cum_weights = torch.cumsum(weights, dim=1)
        
        trend = cum_weighted_x / cum_weights
        
        # 3. Residual 계산 (Seasonal) 
        seasonal = x - trend
        
        return seasonal, trend

    def forward(self, pred, true):
        """
        Section 3.3: Weighted Loss Function 계산 
        
        Args:
            pred (torch.Tensor): 예측값 (Batch, Time, Channels)
            true (torch.Tensor): 실제값 (Batch, Time, Channels)
        """
        # 1. EMA Decomposition 수행
        pred_seasonal, pred_trend = self.ema_decomposition(pred)
        true_seasonal, true_trend = self.ema_decomposition(true)
        
        # 2. 개별 컴포넌트 Loss 계산
        # L_S: Seasonal 성분은 L2 Norm (MSE)
        loss_s = self.mse_loss(pred_seasonal, true_seasonal)
        # L_T: Trend 성분은 L1 Norm (MAE)
        loss_t = self.l1_loss(pred_trend, true_trend)
        
        # 3. Scale Alignment
        # Trend loss의 스케일을 Seasonal loss 수준으로 맞춤
        # stopgrad()는 .detach()로 구현하여 역전파 차단 
        trend_ratio = loss_s / (loss_t.detach() + self.epsilon)
        loss_t_aligned = trend_ratio * loss_t
        
        # 4. 최종 Loss 계산
        loss = self.beta * loss_s + (1 - self.beta) * loss_t_aligned
        
        return loss