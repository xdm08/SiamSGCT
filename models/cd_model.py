import torch
import torch.nn as nn
import torch.nn.functional as F

from .mit_backbone import MiTBackbone
from .gct_module import GCT, TransitionBlock, DifferenceTransformer
from .cpgf import CPGF


class SiamSGCT(nn.Module):
    def __init__(self, num_node_features=128, gcn_hidden=256, transformer_heads=4, num_classes=1):
        super(SiamSGCT, self).__init__()
        
        self.backbone = MiTBackbone(pretrained=True)
        self.feat_dim = 224 
        
        self.gct1 = GCT(in_channels=gcn_hidden, hidden_channels=gcn_hidden, num_heads=transformer_heads)
        self.fusion1 = CPGF(dim=gcn_hidden)
        self.mit_proj = nn.Conv2d(self.feat_dim, gcn_hidden, 1)
        
        self.stage2_transition = TransitionBlock(in_channels=gcn_hidden, out_channels=gcn_hidden, num_heads=transformer_heads)
        self.gct2 = GCT(in_channels=gcn_hidden, hidden_channels=gcn_hidden, num_heads=transformer_heads)
        self.fusion2 = CPGF(dim=gcn_hidden)
        
        self.diff_transformer = DifferenceTransformer(dim=gcn_hidden, num_heads=transformer_heads)
        
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(gcn_hidden + 64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.up_sample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(128 + 32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.up_sample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.raw_fusion_conv = nn.Sequential(
            nn.Conv2d(64 + 3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
        self.aux_head1 = nn.Conv2d(256 + 64, num_classes, kernel_size=1)
        self.aux_head2 = nn.Conv2d(128, num_classes, kernel_size=1)
        
    def forward(self, imgA, imgB, segments, return_aux=False):
        f1_1, f1_2, f1_3 = self.backbone(imgA) 
        f2_1, f2_2, f2_3 = self.backbone(imgB)
        
        f1_3_up = F.interpolate(f1_3, size=f1_2.shape[2:], mode='bilinear', align_corners=True)
        f1_feat = torch.cat([f1_2, f1_3_up], dim=1)
        
        f2_3_up = F.interpolate(f2_3, size=f2_2.shape[2:], mode='bilinear', align_corners=True)
        f2_feat = torch.cat([f2_2, f2_3_up], dim=1)
        
        f1_feat_proj = self.mit_proj(f1_feat)
        f2_feat_proj = self.mit_proj(f2_feat)
        
        x1_gct1 = self.gct1(f1_feat_proj, imgA, segments)
        x2_gct1 = self.gct1(f2_feat_proj, imgB, segments)
        
        x1_fused1 = self.fusion1(f1_feat_proj, x1_gct1)
        x2_fused1 = self.fusion1(f2_feat_proj, x2_gct1)
        
        x1_stage2_in = self.stage2_transition(x1_fused1)
        x2_stage2_in = self.stage2_transition(x2_fused1)
        
        x1_gct2 = self.gct2(x1_stage2_in, imgA, segments)
        x2_gct2 = self.gct2(x2_stage2_in, imgB, segments)
        
        x1_fused2 = self.fusion2(x1_stage2_in, x1_gct2)
        x2_fused2 = self.fusion2(x2_stage2_in, x2_gct2)
        
        diff = torch.abs(x1_fused2 - x2_fused2)
        diff_out = self.diff_transformer(diff)
        
        backbone_diff = torch.abs(f1_2 - f2_2)
        
        x = torch.cat([diff_out, backbone_diff], dim=1)
        x = self.up_conv1(x)
        
        aux1 = self.aux_head1(torch.cat([diff_out, backbone_diff], dim=1))
        aux2 = self.aux_head2(x)
        
        x = self.up_sample1(x)
        
        backbone_diff_1 = torch.abs(f1_1 - f2_1)
        x = torch.cat([x, backbone_diff_1], dim=1)
        x = self.up_conv2(x)
        
        x = self.up_sample2(x)
        
        raw_diff = torch.abs(imgA - imgB)
        x = torch.cat([x, raw_diff], dim=1)
        
        x = self.raw_fusion_conv(x)
        out = self.final_conv(x)
        
        if return_aux:
            return out, aux1, aux2
        return out