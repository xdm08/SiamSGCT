import torch


def calculate_metrics(preds, targets):
    """
    计算变化检测评价指标
    Args:
        preds: 模型预测输出 (B, 1, H, W)
        targets: 真实标签 (B, 1, H, W)
    Returns:
        precision, recall, f1, iou, oa
    """
    preds = (preds > 0).float()
    targets = (targets > 0.5).float()
    
    tp = (preds * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    
    eps = 1e-6
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return precision, recall, f1, iou, oa