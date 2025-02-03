import torch
import polars as pl
from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import random

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform  

    def __call__(self, x):
        q = self.base_transform(x)  
        k = self.base_transform(x)  
        return q, k  

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR and MoCo v2"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma  

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])  
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))  

def get_moco_v2_augmentations():
    """
    Returns the MoCo v2 augmentation pipeline.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  
        transforms.RandomGrayscale(p=0.2),  
        GaussianBlur(sigma=[0.1, 2.0]),  # Use custom GaussianBlur
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

class MoCoTileDataset(Dataset):
    """
    MoCo v2 Dataset for histopathology images.
    Returns two augmented versions of each image.
    """
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (str): Path to the CSV file containing tile paths.
            transform (callable): Augmentation pipeline.
        """
        self.metadata = pl.read_csv(csv_path).select("tile_path")
        self.tile_paths = self.metadata["tile_path"].to_list()
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = Path(self.tile_paths[idx])
        image = Image.open(tile_path).convert("RGB")  # Open as RGB
        
        q, k = self.transform(image)  # Apply MoCo augmentations (TwoCropsTransform)
        return q, k  # Return (query, key) pair
"""
# Initialize MoCo v2 DataLoader
if __name__ == "__main__":
    csv_path = "/home/darya/Histo/Histo_pipeline_csv/tile_path.csv"

    # Wrap MoCo augmentations inside TwoCropsTransform
    train_transform = TwoCropsTransform(get_moco_v2_augmentations())

    # Initialize dataset
    train_dataset = MoCoTileDataset(csv_path=csv_path, transform=train_transform)

    # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

    # Check first batch
    for images_q, images_k in train_loader:
        print(f"Batch size: {images_q.shape}, {images_k.shape}")
        break  # Print only the first batch for verification
"""       