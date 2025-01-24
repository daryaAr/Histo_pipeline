import os
from pathlib import Path
from PIL import Image
import polars as pl  
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistologyDataset(Dataset):
    """
    A custom dataset class for loading histology images and their neighbors.
    """

    def __init__(self, neighbors_csv, transform=None, max_neighbors=8):
        """
        Initialize the dataset.

        Parameters:
        - neighbors_csv: str, path to the CSV file containing neighbors and paths information.
        - transform: torchvision.transforms, augmentations to apply to the images.
        - max_neighbors: int, maximum number of neighbors to consider.
        """
       
        self.data = pl.read_csv(neighbors_csv)
        self.transform = transform
        self.max_neighbors = max_neighbors
        logger.info(f"Dataset initialized with {len(self.data)} tiles and neighbors.")

    def _load_tile(self, tile_path):
        """
        Load a tile from disk.

        Parameters:
        - tile_path: str, path to the tile.

        Returns:
        - PIL.Image: The loaded tile image.
        """
        if tile_path is None or not Path(str(tile_path)).exists():
            raise FileNotFoundError(f"Tile path {tile_path} does not exist.")
        tile_image = Image.open(str(tile_path)).convert("RGB")  
        return tile_image

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return self.data.height  

    def __getitem__(self, idx):
        """
        Load a tile and its neighbors, and apply transformations.

        Parameters:
        - idx: int, index of the tile to retrieve.

        Returns:
        - torch.Tensor: Center tile image tensor.
        - str: Center tile name.
        - str: WSI name the tile belongs to.
        - torch.Tensor: Tensor of neighbors ([max_neighbors, C, H, W]).
        - int: Number of valid neighbors.
        """
        # Extract the row as a tuple and convert it into a dictionary
        row_values = self.data.row(idx)  # Returns a tuple of values
        row = dict(zip(self.data.columns, row_values))  # Combine column names with values

        # Load center tile
        center_tile_path = row["center_tile_path"]
        center_image = self._load_tile(center_tile_path)
        if self.transform:
            center_image = self.transform(center_image)

        # Load neighbors
        neighbors = []
        for i in range(1, self.max_neighbors + 1):
            neighbor_path = row.get(f"neighbor{i}_path", None)
            if neighbor_path is None or not Path(str(neighbor_path)).exists():
                neighbors.append(torch.zeros_like(center_image))  # Zero tensor for missing neighbors
            else:
                neighbor_image = self._load_tile(neighbor_path)
                if self.transform:
                    neighbor_image = self.transform(neighbor_image)
                neighbors.append(neighbor_image)

        neighbors_tensor = torch.stack(neighbors)  # [max_neighbors, C, H, W]
        real_neighbors_count = (neighbors_tensor.sum(dim=(1, 2, 3)) > 0).sum().item()

        return center_image, row["center_tile"], row["wsi"], neighbors_tensor, real_neighbors_count


def get_loader(neighbors_csv, batch_size, augment=True, num_workers=4, pin_memory=True, prefetch_factor=2):
    """
    Initialize a DataLoader for the histology dataset.

    Parameters:
    - neighbors_csv: Path to the CSV file with neighbor and path information.
    - batch_size: Batch size for the DataLoader.
    - augment: Whether to apply augmentations.
    - num_workers: Number of worker processes for data loading.
    - pin_memory: Whether to use pinned memory for faster transfer to GPU.
    - prefetch_factor: Number of batches to prefetch.

    Returns:
    - DataLoader instance.
    """
    logger.info(f"Initializing DataLoader...")

    if augment:
        
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])
        logger.info("Using data augmentation.")
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        logger.info("Using basic transformations (no augmentation).")

    dataset = HistologyDataset(neighbors_csv=neighbors_csv, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True  # Keep workers alive for performance
    )

    return data_loader

"""
# Main function for testing
if __name__ == "__main__":
    neighbors_csv = "/home/darya/Histo_pipeline/updated_neighbor_results.csv"  # Updated CSV file
    batch_size = 32

    logger.info("Testing DataLoader...")

    loader = get_loader(neighbors_csv, batch_size, augment=True, num_workers=4, pin_memory=True)

    for batch_idx, (center_images, center_tile_names, wsi_names, neighbors_tensor, real_neighbors_count) in enumerate(loader):
        logger.info(f"Batch {batch_idx + 1}: Loaded {len(center_tile_names)} center tiles.")
        logger.info(f"Example Center Tile: {center_tile_names[0]}, WSI: {wsi_names[0]}")
        break  # Stop after the first batch for testing
"""       