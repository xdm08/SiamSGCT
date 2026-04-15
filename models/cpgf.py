import torch
import torch.nn as nn

class CPGF(nn.Module):
    """
    Cross-Space Pixel-Graph Collaborative Fusion (CPGF)
    跨空间像素-图协同学习机制
    (原名: PixelGCTFusion)
    """
    def __init__(self, dim):
        super(CPGF, self).__init__()
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_mit, x_gct):
        B, C, H, W = x_mit.shape
        
        # 将空间维度展平为序列 (B, N, C)
        x_mit_flat = x_mit.flatten(2).transpose(1, 2)
        x_gct_flat = x_gct.flatten(2).transpose(1, 2)
        
        # 交叉注意力 (Cross-Attention)
        Q = self.w_q(x_mit_flat)
        K = self.w_k(x_mit_flat)
        V = self.w_v(x_gct_flat)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ V
        
        # 残差连接与前馈网络
        out = self.norm(out + self.mlp(out))
        out = x_mit_flat + out
        
        return out.transpose(1, 2).view(B, C, H, W)