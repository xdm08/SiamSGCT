import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice


class BoundaryLoss(nn.Module):
    def __init__(self, boundary_weight=5.0):
        super(BoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.register_buffer('laplacian_kernel', 
                           torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                      dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        kernel = self.laplacian_kernel.to(target.device)
        boundary = F.conv2d(target, kernel, padding=1).abs() > 0
        weight = 1.0 + self.boundary_weight * boundary.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, weight=weight, reduction='mean')
        return bce