import os
import torch
import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets.levir_cd import LevirCDDataset
from models.cd_model import SiamSGCT
from utils.losses import DiceLoss, BoundaryLoss


def train_model(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    boundary_loss = BoundaryLoss(boundary_weight=5.0)
    
    pbar = tqdm.tqdm(dataloader, desc="Training")
    for imgA, imgB, label, segments in pbar:
        imgA, imgB, label, segments = imgA.to(device), imgB.to(device), label.to(device), segments.to(device)
        
        optimizer.zero_grad()
        
        output, aux1, aux2 = model(imgA, imgB, segments, return_aux=True)
        
        main_loss = boundary_loss(output, label) + dice_loss(output, label)
        
        label_aux1 = F.interpolate(label, size=aux1.shape[2:], mode='nearest')
        label_aux2 = F.interpolate(label, size=aux2.shape[2:], mode='nearest')
        
        aux_loss1 = bce_loss(aux1, label_aux1) + dice_loss(aux1, label_aux1)
        aux_loss2 = bce_loss(aux2, label_aux2) + dice_loss(aux2, label_aux2)
        
        loss = main_loss + 0.4 * aux_loss1 + 0.4 * aux_loss2
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    
    tp_all = 0
    tn_all = 0
    fp_all = 0
    fn_all = 0
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for imgA, imgB, label, segments in tqdm.tqdm(dataloader, desc="Validating"):
            imgA, imgB, label, segments = imgA.to(device), imgB.to(device), label.to(device), segments.to(device)
            
            output = model(imgA, imgB, segments)
            loss = bce_loss(output, label)
            running_loss += loss.item()
            
            preds = (output > 0).float()
            targets = (label > 0.5).float()
            
            tp = (preds * targets).sum().item()
            tn = ((1 - preds) * (1 - targets)).sum().item()
            fp = (preds * (1 - targets)).sum().item()
            fn = ((1 - preds) * targets).sum().item()
            
            tp_all += tp
            tn_all += tn
            fp_all += fp
            fn_all += fn
            
    avg_loss = running_loss / len(dataloader)
    
    eps = 1e-6
    
    precision = tp_all / (tp_all + fp_all + eps)
    recall = tp_all / (tp_all + fn_all + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp_all / (tp_all + fp_all + fn_all + eps)
    oa = (tp_all + tn_all) / (tp_all + tn_all + fp_all + fn_all + eps)
    
    return avg_loss, precision, recall, f1, iou, oa


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 150
    LR = 4e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ROOT_DIR = '/root/autodl-tmp/LEVIR-CD256'
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = open(f'train_log_{timestamp}.txt', 'w', encoding='utf-8')
    def log_print(text):
        print(text)
        log_file.write(str(text) + '\n')
        log_file.flush()
    
    log_print(f"Using device: {DEVICE}")
    
    train_dataset = LevirCDDataset(ROOT_DIR, split='train', n_segments=150)
    val_dataset = LevirCDDataset(ROOT_DIR, split='val', n_segments=150)
    
    if len(val_dataset) == 0:
        log_print("Validation set empty, using subset of train for val.")
        total = len(train_dataset)
        train_size = int(0.9 * total)
        val_size = total - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = SiamSGCT().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_f1 = 0.0
    for epoch in range(EPOCHS):
        log_print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_model(model, train_loader, optimizer, DEVICE)
        val_loss, precision, recall, f1, iou, oa = evaluate(model, val_loader, DEVICE)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        log_print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        log_print(f"Metrics: Precision: {precision*100:.2f} | Recall: {recall*100:.2f} | F1: {f1*100:.2f} | IoU: {iou*100:.2f} | OA: {oa*100:.2f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'siamsgct_best.pth')
            log_print(f"✓ Best model saved! F1: {f1*100:.2f}")
            
    log_file.close()