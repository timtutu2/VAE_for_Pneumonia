"""
Dataset loader for Chest X-ray images
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ChestXrayDataset(Dataset):
    """
    Dataset class for Chest X-ray images
    Supports both train and test splits
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory containing train/test folders
            split: 'train' or 'test'
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Collect all image paths
        self.image_paths = []
        split_dir = os.path.join(root_dir, split)
        
        if os.path.exists(split_dir):
            # Iterate through NORMAL and PNEUMONIA folders
            for category in ['NORMAL', 'PNEUMONIA']:
                category_dir = os.path.join(split_dir, category)
                if os.path.exists(category_dir):
                    for img_name in os.listdir(category_dir):
                        if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                            self.image_paths.append(os.path.join(category_dir, img_name))
        
        print(f"Loaded {len(self.image_paths)} images from {split} set")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image and convert to grayscale
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_data_loaders(data_dir, batch_size=32, image_size=224, num_workers=4):
    """
    Create train and test data loaders
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size for training
        image_size: Size to resize images to
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = ChestXrayDataset(data_dir, split='train', transform=transform)
    test_dataset = ChestXrayDataset(data_dir, split='test', transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

