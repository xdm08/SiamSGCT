import os
import torch
import random
from skimage.segmentation import slic
from skimage import io
from torchvision import transforms


class WHUCDDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None, n_segments=100):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.n_segments = n_segments
        
        self.dir_A = os.path.join(root_dir, 'A')
        self.dir_B = os.path.join(root_dir, 'B')
        self.dir_label = os.path.join(root_dir, 'label')
        
        all_files = os.listdir(self.dir_A)
        
        # 检查list目录是否存在
        list_dir = os.path.join(self.root_dir, 'list')
        
        # 优先使用list目录中的txt文件进行划分
        if os.path.exists(list_dir):
            train_list_path = os.path.join(list_dir, 'train.txt')
            val_list_path = os.path.join(list_dir, 'val.txt')
            test_list_path = os.path.join(list_dir, 'test.txt')
            
            if os.path.exists(train_list_path) and os.path.exists(val_list_path) and os.path.exists(test_list_path):
                # 从txt文件读取划分
                with open(train_list_path, 'r') as f:
                    train_files = [line.strip() for line in f.readlines() if line.strip()]
                with open(val_list_path, 'r') as f:
                    val_files = [line.strip() for line in f.readlines() if line.strip()]
                with open(test_list_path, 'r') as f:
                    test_files = [line.strip() for line in f.readlines() if line.strip()]
                
                self._train_files = train_files
                self._val_files = val_files
                self._test_files = test_files
            else:
                # 如果没有txt文件，回退到文件名前缀方式
                train_files = [f for f in all_files if f.startswith('train_')]
                val_files = [f for f in all_files if f.startswith('val_')]
                test_files = [f for f in all_files if f.startswith('test_')]
                
                # 如果没有预定义的分割文件，自动随机划分
                if len(train_files) == 0 and len(val_files) == 0 and len(test_files) == 0:
                    print("Warning: No predefined split files found. Automatically splitting dataset 80% train, 10% val, 10% test.")
                    
                    random.shuffle(all_files)
                    total_files = len(all_files)
                    train_size = int(0.8 * total_files)
                    val_size = int(0.1 * total_files)
                    test_size = total_files - train_size - val_size
                    
                    train_files = all_files[:train_size]
                    val_files = all_files[train_size:train_size + val_size]
                    test_files = all_files[train_size + val_size:]
                
                self._train_files = train_files
                self._val_files = val_files
                self._test_files = test_files
        else:
            # 如果没有list目录，使用文件名前缀方式
            train_files = [f for f in all_files if f.startswith('train_')]
            val_files = [f for f in all_files if f.startswith('val_')]
            test_files = [f for f in all_files if f.startswith('test_')]
            
            if len(train_files) == 0 and len(val_files) == 0 and len(test_files) == 0:
                print("Warning: No predefined split files found. Automatically splitting dataset 80% train, 10% val, 10% test.")
                
                random.shuffle(all_files)
                total_files = len(all_files)
                train_size = int(0.8 * total_files)
                val_size = int(0.1 * total_files)
                test_size = total_files - train_size - val_size
                
                train_files = all_files[:train_size]
                val_files = all_files[train_size:train_size + val_size]
                test_files = all_files[train_size + val_size:]
            
            self._train_files = train_files
            self._val_files = val_files
            self._test_files = test_files
        
        # 根据请求的split选择文件
        if split == 'train':
            self.files = self._train_files
        elif split == 'val':
            self.files = self._val_files
        elif split == 'test':
            self.files = self._test_files
        else:
            self.files = all_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        path_A = os.path.join(self.dir_A, filename)
        path_B = os.path.join(self.dir_B, filename)
        path_label = os.path.join(self.dir_label, filename)
        
        img_A = io.imread(path_A)
        img_B = io.imread(path_B)
        label = io.imread(path_label)
        
        segments = slic(img_A, n_segments=self.n_segments, compactness=10, sigma=1, start_label=0)
        
        img_A_tensor = torch.from_numpy(img_A.transpose((2, 0, 1))).float() / 255.0
        img_B_tensor = torch.from_numpy(img_B.transpose((2, 0, 1))).float() / 255.0
        
        if len(label.shape) == 3:
            label = label[:, :, 0]
        label_tensor = torch.from_numpy(label).float() / 255.0
        label_tensor = (label_tensor > 0.5).float().unsqueeze(0)
        
        segments_tensor = torch.from_numpy(segments).long().unsqueeze(0)

        if self.split == 'train':
            if torch.rand(1) > 0.5:
                img_A_tensor = transforms.functional.hflip(img_A_tensor)
                img_B_tensor = transforms.functional.hflip(img_B_tensor)
                label_tensor = transforms.functional.hflip(label_tensor)
                segments_tensor = transforms.functional.hflip(segments_tensor)

            if torch.rand(1) > 0.5:
                img_A_tensor = transforms.functional.vflip(img_A_tensor)
                img_B_tensor = transforms.functional.vflip(img_B_tensor)
                label_tensor = transforms.functional.vflip(label_tensor)
                segments_tensor = transforms.functional.vflip(segments_tensor)

            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                img_A_tensor = torch.rot90(img_A_tensor, k, [1, 2])
                img_B_tensor = torch.rot90(img_B_tensor, k, [1, 2])
                label_tensor = torch.rot90(label_tensor, k, [1, 2])
                segments_tensor = torch.rot90(segments_tensor, k, [1, 2])

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_A_tensor = normalize(img_A_tensor)
        img_B_tensor = normalize(img_B_tensor)
        
        segments_tensor = segments_tensor.squeeze(0)
            
        return img_A_tensor, img_B_tensor, label_tensor, segments_tensor