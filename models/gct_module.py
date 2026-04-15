import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=4):
        super(GCT, self).__init__()
        self.hidden_channels = hidden_channels
        
        self.proj_in = nn.Linear(in_channels, hidden_channels)
        
        self.gcn = GCNConv(hidden_channels, hidden_channels)
        self.relu = nn.ReLU()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=num_heads, 
                                                 dim_feedforward=hidden_channels*2, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def compute_association_matrix(self, img, segments):
        B, _, H, W = img.shape
        N_pixels = H * W
        device = img.device
        
        img_flat = img.view(B, 3, -1)
        seg_flat = segments.view(B, -1)
        
        O_list = []
        P_list = []
        
        for b in range(B):
            seg_b = seg_flat[b]
            unique_segs = torch.unique(seg_b)
            num_segs = len(unique_segs)
            
            seg_remapped = torch.searchsorted(unique_segs, seg_b)
            
            grad_dummy = torch.zeros(num_segs, 3, device=device)
            grad_dummy.index_add_(0, seg_remapped, img_flat[b].permute(1, 0))
            
            counts = torch.zeros(num_segs, device=device)
            counts.index_add_(0, seg_remapped, torch.ones(N_pixels, device=device))
            
            means = grad_dummy / (counts.unsqueeze(1) + 1e-8)
            
            pixel_means = means[seg_remapped]
            pixel_vals = img_flat[b].permute(1, 0)
            
            sim = F.cosine_similarity(pixel_vals, pixel_means, dim=1)
            sim = torch.clamp(sim, 0, 1)
            
            indices = torch.stack([seg_remapped, torch.arange(N_pixels, device=device)], dim=0)
            values = sim
            
            O_mat = torch.sparse_coo_tensor(indices, values, (num_segs, N_pixels)).to_dense()
            
            norm_factors = torch.sum(O_mat ** 2, dim=1)
            inv_norm = 1.0 / (norm_factors + 1e-8)
            
            P_mat = O_mat * inv_norm.unsqueeze(1)
            
            O_list.append(O_mat)
            P_list.append(P_mat)
            
        return O_list, P_list

    def forward(self, x, img, segments):
        B, C, H, W = x.shape
        
        img_small = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=True)
        seg_small = F.interpolate(segments.unsqueeze(1).float(), size=(H, W), mode='nearest').long().squeeze(1)
        
        O_list, P_list = self.compute_association_matrix(img_small, seg_small)
        
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        
        outputs = []
        
        for b in range(B):
            P = P_list[b]
            O = O_list[b]
            
            feat = x_flat[b]
            
            node_feat = torch.mm(P, feat)
            node_feat = self.proj_in(node_feat)
            
            seg_map = seg_small[b]
            unique_segs = torch.unique(seg_map)
            seg_map_remapped = torch.searchsorted(unique_segs, seg_map)
            
            right_mask = seg_map_remapped[:, :-1] != seg_map_remapped[:, 1:]
            edges_r = torch.stack([seg_map_remapped[:, :-1][right_mask], seg_map_remapped[:, 1:][right_mask]], dim=0)
            down_mask = seg_map_remapped[:-1, :] != seg_map_remapped[1:, :]
            edges_d = torch.stack([seg_map_remapped[:-1, :][down_mask], seg_map_remapped[1:, :][down_mask]], dim=0)
            edge_index = torch.cat([edges_r, edges_d], dim=1)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = torch.unique(edge_index, dim=1)
            
            node_feat = self.relu(self.gcn(node_feat, edge_index))
            
            node_feat_in = node_feat.unsqueeze(1)
            node_feat_out = self.transformer(node_feat_in).squeeze(1)
            
            pixel_out = torch.mm(O.t(), node_feat_out)
            
            outputs.append(pixel_out)
            
        x_out = torch.stack(outputs, dim=0)
        x_out = x_out.permute(0, 2, 1).view(B, -1, H, W)
        
        return x_out


class DifferenceTransformer(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(DifferenceTransformer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*2, dropout=0.1),
            num_layers=1
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, dim) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        
        if x_flat.size(1) == self.pos_embed.size(1):
             x_flat = x_flat + self.pos_embed
             
        x_in = x_flat.permute(1, 0, 2)
        x_out = self.transformer(x_in)
        x_out = x_out.permute(1, 0, 2)
        
        return x_out.transpose(1, 2).view(B, C, H, W)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.):
        super(TransitionBlock, self).__init__()
        from .mit_backbone import Block
        
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(out_channels)
        
        self.block = Block(
            dim=out_channels, num_heads=num_heads, mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, 
            drop_path=drop_path, sr_ratio=1
        )
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.downsample(x)
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        
        x = self.block(x, H, W)
        x = self.norm2(x)
        
        return x.transpose(1, 2).view(B, C, H, W)