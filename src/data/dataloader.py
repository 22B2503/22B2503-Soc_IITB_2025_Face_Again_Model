"""
PyTorch DataLoader for Face Aging Model
Handles dataset loading, batching, and augmentation
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
import json
import random
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

class FaceAgingDataset(Dataset):
    """PyTorch Dataset for face aging model training"""
    
    def __init__(self, 
                 data_dir: str,
                 metadata_path: Optional[str] = None,
                 transform: Optional[A.Compose] = None,
                 target_size: Tuple[int, int] = (256, 256),
                 age_bins: Optional[List[int]] = None):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing face images
            metadata_path: Path to metadata JSON file
            transform: Albumentations transform pipeline
            target_size: Target image size (width, height)
            age_bins: Age bin boundaries for classification
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.age_bins = age_bins or [0, 30, 60, 100]  # Default age bins
        
        # Load metadata if provided
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.image_paths = self.metadata.get('image_paths', [])
            self.age_labels = self.metadata.get('age_labels', {})
        else:
            # Auto-discover images and create labels
            self.image_paths = self._discover_images()
            self.age_labels = self._create_age_labels()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
        
        print(f"Loaded {len(self.image_paths)} images")
        print(f"Age distribution: {self._get_age_distribution()}")
    
    def _discover_images(self) -> List[str]:
        """Discover all image files in the data directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.lower().endswith(ext):
                        image_paths.append(os.path.join(root, file))
        
        return image_paths
    
    def _create_age_labels(self) -> Dict[str, int]:
        """Create age labels based on filename patterns"""
        age_labels = {}
        
        for image_path in self.image_paths:
            filename = os.path.basename(image_path).lower()
            
            # Extract age from filename patterns
            if 'young' in filename or 'age_20' in filename or 'age_25' in filename:
                age_labels[image_path] = 25
            elif 'middle' in filename or 'age_40' in filename or 'age_50' in filename:
                age_labels[image_path] = 45
            elif 'elderly' in filename or 'age_60' in filename or 'age_70' in filename:
                age_labels[image_path] = 65
            else:
                # Default age if no pattern matches
                age_labels[image_path] = 35
        
        return age_labels
    
    def _get_age_distribution(self) -> Dict[int, int]:
        """Get age distribution in the dataset"""
        age_counts = {}
        for age in self.age_labels.values():
            age_counts[age] = age_counts.get(age, 0) + 1
        return age_counts
    
    def _get_default_transforms(self) -> A.Compose:
        """Get default transformation pipeline"""
        return A.Compose([
            A.Resize(self.target_size[1], self.target_size[0]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    
    def _age_to_bin(self, age: int) -> int:
        """Convert age to age bin index"""
        for i, bin_boundary in enumerate(self.age_bins[1:], 1):
            if age < bin_boundary:
                return i - 1
        return len(self.age_bins) - 2  # Last bin
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset"""
        image_path = self.image_paths[idx]
        
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a placeholder image
            image = np.zeros((*self.target_size, 3), dtype=np.uint8)
        
        # Get age label
        age = self.age_labels.get(image_path, 35)
        age_bin = self._age_to_bin(age)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
        else:
            # Basic preprocessing without augmentation
            image = cv2.resize(image, self.target_size)
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.from_numpy(image)
        
        return {
            'image': image_tensor,
            'age': torch.tensor(age, dtype=torch.float32),
            'age_bin': torch.tensor(age_bin, dtype=torch.long),
            'image_path': image_path
        }

class FaceAgingDataLoader:
    """DataLoader wrapper for face aging dataset"""
    
    def __init__(self, 
                 data_dir: str,
                 metadata_path: Optional[str] = None,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 target_size: Tuple[int, int] = (256, 256),
                 train_split: float = 0.8,
                 val_split: float = 0.1,
                 test_split: float = 0.1):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing face images
            metadata_path: Path to metadata JSON file
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            target_size: Target image size
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.target_size = target_size
        
        # Create datasets
        self.train_dataset, self.val_dataset, self.test_dataset = self._create_datasets(
            data_dir, metadata_path, target_size, train_split, val_split, test_split
        )
        
        # Create data loaders
        self.train_loader = self._create_data_loader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_data_loader(self.val_dataset, shuffle=False)
        self.test_loader = self._create_data_loader(self.test_dataset, shuffle=False)
    
    def _create_datasets(self, 
                        data_dir: str,
                        metadata_path: Optional[str],
                        target_size: Tuple[int, int],
                        train_split: float,
                        val_split: float,
                        test_split: float) -> Tuple[FaceAgingDataset, FaceAgingDataset, FaceAgingDataset]:
        """Create train, validation, and test datasets"""
        
        # Create full dataset
        full_dataset = FaceAgingDataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            target_size=target_size
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Create splits
        train_indices, val_indices, test_indices = torch.utils.data.random_split(
            range(total_size), [train_size, val_size, test_size]
        )
        
        # Create datasets for each split
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices.indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices.indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_data_loader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> DataLoader:
        """Create a DataLoader for the given dataset"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle  # Drop last batch during training
        )
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test data loaders"""
        return self.train_loader, self.val_loader, self.test_loader
    
    def visualize_batch(self, batch: Dict[str, torch.Tensor], num_samples: int = 8):
        """Visualize a batch of images"""
        images = batch['image'][:num_samples]
        ages = batch['age'][:num_samples]
        
        # Denormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
        
        # Convert to numpy
        images = images.permute(0, 2, 3, 1).numpy()
        
        # Create subplot
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(images))):
            axes[i].imshow(images[i])
            axes[i].set_title(f'Age: {ages[i].item():.0f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

class AgeConditionalDataset(Dataset):
    """Dataset for age-conditional generation (for GANs)"""
    
    def __init__(self, 
                 data_dir: str,
                 metadata_path: Optional[str] = None,
                 transform: Optional[A.Compose] = None,
                 target_size: Tuple[int, int] = (256, 256),
                 target_ages: List[int] = None):
        """
        Initialize age-conditional dataset
        
        Args:
            data_dir: Directory containing face images
            metadata_path: Path to metadata JSON file
            transform: Albumentations transform pipeline
            target_size: Target image size
            target_ages: List of target ages for conditioning
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.target_ages = target_ages or [25, 45, 65]
        
        # Load base dataset
        self.base_dataset = FaceAgingDataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            transform=transform,
            target_size=target_size
        )
        
        # Create age conditioning vectors
        self.age_vectors = self._create_age_vectors()
    
    def _create_age_vectors(self) -> torch.Tensor:
        """Create one-hot age conditioning vectors"""
        num_ages = len(self.target_ages)
        age_vectors = torch.zeros(num_ages, num_ages)
        for i in range(num_ages):
            age_vectors[i, i] = 1.0
        return age_vectors
    
    def __len__(self) -> int:
        return len(self.base_dataset) * len(self.target_ages)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample with age conditioning"""
        base_idx = idx // len(self.target_ages)
        age_idx = idx % len(self.target_ages)
        
        # Get base sample
        base_sample = self.base_dataset[base_idx]
        
        # Get target age
        target_age = self.target_ages[age_idx]
        age_vector = self.age_vectors[age_idx]
        
        return {
            'image': base_sample['image'],
            'original_age': base_sample['age'],
            'target_age': torch.tensor(target_age, dtype=torch.float32),
            'age_condition': age_vector,
            'image_path': base_sample['image_path']
        }

def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory function to create data loaders from config"""
    
    data_loader = FaceAgingDataLoader(
        data_dir=config['data_dir'],
        metadata_path=config.get('metadata_path'),
        batch_size=config['batch_size'],
        shuffle=config.get('shuffle', True),
        num_workers=config.get('num_workers', 4),
        target_size=tuple(config['target_size']),
        train_split=config.get('train_split', 0.8),
        val_split=config.get('val_split', 0.1),
        test_split=config.get('test_split', 0.1)
    )
    
    return data_loader.get_data_loaders()

def main():
    """Main function to test the data loader"""
    
    # Example configuration
    config = {
        'data_dir': 'data/processed',
        'metadata_path': 'data/processed/dataset_metadata.json',
        'batch_size': 16,
        'target_size': [256, 256],
        'num_workers': 2
    }
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test a batch
        for batch in train_loader:
            print(f"Batch keys: {batch.keys()}")
            print(f"Image shape: {batch['image'].shape}")
            print(f"Age shape: {batch['age'].shape}")
            print(f"Age bin shape: {batch['age_bin'].shape}")
            
            # Visualize batch
            data_loader = FaceAgingDataLoader(**config)
            data_loader.visualize_batch(batch)
            break
            
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        print("Make sure you have processed data in the data/processed directory")

if __name__ == "__main__":
    main() 