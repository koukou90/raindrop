import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving average block to highlight trend."""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        trend = x.permute(0, 2, 1)
        return trend


class SeriesDecomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class LinearBaseline(nn.Module):
    """
    DLinear 官方思路适配版（保留 BaselineLinear 名称以兼容现有命令）。
    输入:  conc/vel/phys
    输出:  RainRate 未来 pred_len 步
    """

    def __init__(
        self,
        args=None,
        seq_len=10,
        conc_dim=32,
        vel_dim=32,
        phys_dim=5,
        pred_len=5,
        moving_avg_kernel=9,
    ):
        super().__init__()

        if args is not None:
            seq_len = getattr(args, 'seq_len', seq_len)
            conc_dim = getattr(args, 'conc_dim', conc_dim)
            vel_dim = getattr(args, 'vel_dim', vel_dim)
            phys_dim = getattr(args, 'phys_dim', phys_dim)
            pred_len = getattr(args, 'pred_len', pred_len)

        self.seq_len = seq_len
        self.pred_len = pred_len
        channels = conc_dim + vel_dim + phys_dim
        self.target_idx = conc_dim + vel_dim  # phys 的 RainRate 列

        kernel = min(moving_avg_kernel, seq_len if seq_len % 2 == 1 else max(1, seq_len - 1))
        if kernel % 2 == 0:
            kernel = max(1, kernel - 1)
        self.decomp = SeriesDecomp(kernel_size=max(1, kernel))

        # official 默认 individual=False，共享线性层
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, conc, vel, phys):
        x = torch.cat([conc, vel, phys], dim=-1)  # (B, L, C)

        # instance normalization（与 refs 保持一致）
        means = x.mean(1, keepdim=True).detach()
        x_norm = x - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_norm = x_norm / stdev

        seasonal, trend = self.decomp(x_norm)
        seasonal = seasonal.permute(0, 2, 1)  # (B, C, L)
        trend = trend.permute(0, 2, 1)        # (B, C, L)

        seasonal_out = self.linear_seasonal(seasonal)  # (B, C, H)
        trend_out = self.linear_trend(trend)           # (B, C, H)
        pred_all = (seasonal_out + trend_out).permute(0, 2, 1)  # (B, H, C)

        # de-normalization
        pred_all = pred_all * stdev + means

        # 只保留 RainRate 目标变量
        return pred_all[:, :, self.target_idx]
