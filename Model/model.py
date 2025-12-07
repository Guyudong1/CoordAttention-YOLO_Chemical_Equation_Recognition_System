import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    """
    YOLO11-compatible Transformer layer.
    """

    def __init__(self, channels, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.channels = channels
        self.fuse = False  # ★ 必须禁用 stride 推断中的 fuse，否则会报维度错

        # TransformerEncoderLayer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )

        # 位置编码
        self.pos_encoding = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # ★★ 避免 YOLO 构建阶段出错（dummy 输入是 3×s×s）
        if x.shape[1] != self.channels:
            return x

        B, C, H, W = x.shape

        x = x + self.pos_encoding(x)

        # reshape for transformer
        x_seq = x.flatten(2).permute(2, 0, 1)

        x_out = self.transformer(x_seq)

        x_out = x_out.permute(1, 2, 0).view(B, C, H, W)
        return x_out
