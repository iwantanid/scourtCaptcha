import os
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np

class CaptchaDataset(Dataset):
    def __init__(self, base_dir=None, transform=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        self.training_dir = os.path.join(base_dir, "TrainingData")
        self.answer_key_path = os.path.join(base_dir, "answer_key.txt")
        self.transform = transform
        
        self.samples = []
        
        # Load the answer_key.txt
        if not os.path.exists(self.answer_key_path):
            raise FileNotFoundError(f"answer_key.txt not found at {self.answer_key_path}!")
            
        with open(self.answer_key_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":")
                filename = parts[0].strip()
                label_str = parts[1].strip()
                
                # We need exactly 6 digits
                assert len(label_str) == 6 and label_str.isdigit(), f"Invalid label: {label_str}"
                
                # Target tensor of labels: e.g. [6, 2, 9, 4, 7, 4]
                label_tensor = torch.tensor([int(c) for c in label_str], dtype=torch.long)
                
                img_path = os.path.join(self.training_dir, filename)
                self.samples.append((img_path, label_tensor, label_str))
                
        print(f"Dataset successfully loaded with {len(self.samples)} captcha samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_tensor, label_str = self.samples[idx]
        
        # Open image as grayscale
        with Image.open(img_path) as img:
            img = img.convert("L")
            # Resize to a standard dimension
            img = img.resize((120, 40), Image.Resampling.BILINEAR)
            
            # Convert to numpy array and normalize
            img_arr = np.array(img, dtype=np.float32) / 255.0
            
            # Convert to PyTorch Tensor (channel, height, width) -> (1, 40, 120)
            img_tensor = torch.tensor(img_arr).unsqueeze(0)
            
        return img_tensor, label_tensor, label_str

class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        img_tensor, label_tensor, label_str = self.subset[idx]
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, label_tensor, label_str
        
    def __len__(self):
        return len(self.subset)

def get_train_val_loaders(base_dir=None, batch_size=64, train_ratio=0.9, seed=42):
    import torchvision.transforms as transforms
    
    dataset = CaptchaDataset(base_dir=base_dir)
    
    # Deterministic split
    generator = torch.Generator().manual_seed(seed)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    raw_train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Data Augmentation only for training set:
    # - Slight rotation (+/- 5 degrees)
    # - Slight translation (+/- 5% in x and y directions)
    # - Slight scaling (95% to 105%)
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
    ])
    
    train_set = DatasetWrapper(raw_train_set, transform=train_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"DataLoader split: {train_size} train samples (augmented), {val_size} validation samples.")
    return train_loader, val_loader, dataset

