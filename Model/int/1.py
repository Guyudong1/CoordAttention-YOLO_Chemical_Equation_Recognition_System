import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, dim, nhead, dim_feedforward, dropout):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
    def forward(self, x):
        # x: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0,2,1)
        x = self.encoder(x)
        x = x.permute(0,2,1).view(B,C,H,W)
        return x
