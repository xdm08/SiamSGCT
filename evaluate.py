import argparse
import torch
from torch.utils.data import DataLoader

from datasets.levir_cd import LevirCDDataset
from models.cd_model import SiamSGCT


def main():
    parser = argparse.ArgumentParser(description='Evaluate Change Detection Model')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--root_dir', type=str, default='/root/autodl-tmp/LEVIR-CD256', help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    test_dataset = LevirCDDataset(args.root_dir, split='test', n_segments=150)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = SiamSGCT().to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.eval()
    
    from train import evaluate
    
    val_loss, precision, recall, f1, iou, oa = evaluate(model, test_loader, DEVICE)
    
    print("\n===== Test Results =====")
    print(f"Loss:      {val_loss:.4f}")
    print(f"Precision: {precision*100:.2f} %")
    print(f"Recall:    {recall*100:.2f} %")
    print(f"F1 Score:  {f1*100:.2f} %")
    print(f"IoU:       {iou*100:.2f} %")
    print(f"OA:        {oa*100:.2f} %")
    print("=======================")


if __name__ == "__main__":
    main()