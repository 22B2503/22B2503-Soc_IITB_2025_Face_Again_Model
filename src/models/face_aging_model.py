import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
import os
from typing import Tuple, Optional, List
import math


class FaceAgingModel(nn.Module):
    """
    Complete Face Aging Model combining CAAE with modern deep learning techniques
    """
    
    def __init__(self, 
                 input_size: int = 128,
                 latent_dim: int = 100,
                 age_embedding_dim: int = 64,
                 num_age_groups: int = 10):
        super(FaceAgingModel, self).__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.age_embedding_dim = age_embedding_dim
        self.num_age_groups = num_age_groups
        
        # Age embedding layer
        self.age_embedding = nn.Embedding(num_age_groups, age_embedding_dim)
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Decoder
        self.decoder = self._build_decoder()
        
        # Discriminator
        self.discriminator = self._build_discriminator()
        
        # Age classifier
        self.age_classifier = self._build_age_classifier()
        
    def _build_encoder(self) -> nn.Module:
        """Build the encoder network"""
        return nn.Sequential(
            # Input: (batch_size, 3, 128, 128)
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(1024),
            
            nn.Conv2d(1024, self.latent_dim, 4, 1, 0, bias=False),  # 1x1
            nn.Tanh()
        )
    
    def _build_decoder(self) -> nn.Module:
        """Build the decoder network"""
        return nn.Sequential(
            # Input: (batch_size, latent_dim + age_embedding_dim, 1, 1)
            nn.ConvTranspose2d(self.latent_dim + self.age_embedding_dim, 1024, 4, 1, 0, bias=False),  # 4x4
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # 8x8
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 16x16
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 32x32
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 64x64
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # 128x128
            nn.Tanh()
        )
    
    def _build_discriminator(self) -> nn.Module:
        """Build the discriminator network"""
        return nn.Sequential(
            # Input: (batch_size, 3, 128, 128)
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # 1x1
            nn.Sigmoid()
        )
    
    def _build_age_classifier(self) -> nn.Module:
        """Build the age classifier network"""
        return nn.Sequential(
            # Input: (batch_size, 3, 128, 128)
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512),
            
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_age_groups),
            nn.Softmax(dim=1)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image to latent representation"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor, age_labels: torch.Tensor) -> torch.Tensor:
        """Decode latent representation with age conditioning"""
        batch_size = z.size(0)
        
        # Get age embeddings
        age_embeddings = self.age_embedding(age_labels)  # (batch_size, age_embedding_dim)
        age_embeddings = age_embeddings.view(batch_size, -1, 1, 1)  # (batch_size, age_embedding_dim, 1, 1)
        
        # Concatenate latent code with age embeddings
        z_combined = torch.cat([z, age_embeddings], dim=1)  # (batch_size, latent_dim + age_embedding_dim, 1, 1)
        
        # Decode
        return self.decoder(z_combined)
    
    def forward(self, x: torch.Tensor, age_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the complete model"""
        # Encode
        z = self.encode(x)
        
        # Decode with age conditioning
        reconstructed = self.decode(z, age_labels)
        
        # Discriminator output
        real_validity = self.discriminator(x)
        fake_validity = self.discriminator(reconstructed)
        
        # Age classification
        age_pred = self.age_classifier(reconstructed)
        
        return reconstructed, real_validity, fake_validity, age_pred


class FaceDataset(Dataset):
    """Custom dataset for face images with age labels"""
    
    def __init__(self, 
                 data_dir: str,
                 transform=None,
                 age_range: Tuple[int, int] = (0, 100),
                 image_size: int = 128):
        self.data_dir = data_dir
        self.transform = transform
        self.age_range = age_range
        self.image_size = image_size
        
        # Load image paths and age labels
        self.image_paths, self.age_labels = self._load_data()
        
    def _load_data(self) -> Tuple[List[str], List[int]]:
        """Load image paths and age labels from directory"""
        image_paths = []
        age_labels = []
        
        # Walk through the data directory
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Extract age from filename or directory structure
                    # Assuming format: age_XX_filename.jpg or directory structure
                    age = self._extract_age_from_path(os.path.join(root, file))
                    
                    if self.age_range[0] <= age <= self.age_range[1]:
                        image_paths.append(os.path.join(root, file))
                        age_labels.append(age)
        
        return image_paths, age_labels
    
    def _extract_age_from_path(self, filepath: str) -> int:
        """Extract age from file path or filename"""
        # Try to extract age from filename
        filename = os.path.basename(filepath)
        
        # Look for age pattern in filename
        import re
        age_match = re.search(r'(\d{1,2})', filename)
        if age_match:
            return int(age_match.group(1))
        
        # Look for age in directory structure
        dir_parts = filepath.split(os.sep)
        for part in dir_parts:
            age_match = re.search(r'(\d{1,2})', part)
            if age_match:
                return int(age_match.group(1))
        
        # Default age if not found
        return 25
    
    def _age_to_group(self, age: int) -> int:
        """Convert age to age group index"""
        age_groups = [
            (0, 10), (11, 20), (21, 30), (31, 40), 
            (41, 50), (51, 60), (61, 70), (71, 80), 
            (81, 90), (91, 100)
        ]
        
        for i, (min_age, max_age) in enumerate(age_groups):
            if min_age <= age <= max_age:
                return i
        
        return 4  # Default to middle age group
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image = image * 2 - 1  # Normalize to [-1, 1]
        
        # Get age group
        age = self.age_labels[idx]
        age_group = self._age_to_group(age)
        
        return image, age_group


class FaceAgingTrainer:
    """Trainer class for the face aging model"""
    
    def __init__(self, 
                 model: FaceAgingModel,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Initialize optimizers
        self.encoder_optimizer = torch.optim.Adam(
            list(self.model.encoder.parameters()) + 
            list(self.model.decoder.parameters()) + 
            list(self.model.age_embedding.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        
        self.discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=0.0002, betas=(0.5, 0.999)
        )
        
        self.age_classifier_optimizer = torch.optim.Adam(
            self.model.age_classifier.parameters(),
            lr=0.0001, betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.reconstruction_loss = nn.L1Loss()
        self.age_classification_loss = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'age_classifier_loss': [],
            'reconstruction_loss': []
        }
    
    def train_step(self, 
                   real_images: torch.Tensor, 
                   age_labels: torch.Tensor,
                   target_ages: torch.Tensor) -> dict:
        """Single training step"""
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        age_labels = age_labels.to(self.device)
        target_ages = target_ages.to(self.device)
        
        # Ground truth labels
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)
        
        # ---------------------
        # Train Generator
        # ---------------------
        self.encoder_optimizer.zero_grad()
        self.age_classifier_optimizer.zero_grad()
        
        # Generate reconstructed images with target ages
        z = self.model.encode(real_images)
        fake_images = self.model.decode(z, target_ages)
        
        # Generator losses
        fake_validity = self.model.discriminator(fake_images)
        g_loss = self.adversarial_loss(fake_validity, valid)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(fake_images, real_images)
        
        # Age classification loss
        age_pred = self.model.age_classifier(fake_images)
        age_loss = self.age_classification_loss(age_pred, target_ages)
        
        # Total generator loss
        total_g_loss = g_loss + 10 * recon_loss + age_loss
        
        total_g_loss.backward()
        self.encoder_optimizer.step()
        self.age_classifier_optimizer.step()
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        self.discriminator_optimizer.zero_grad()
        
        # Real images
        real_validity = self.model.discriminator(real_images)
        d_real_loss = self.adversarial_loss(real_validity, valid)
        
        # Fake images
        fake_validity = self.model.discriminator(fake_images.detach())
        d_fake_loss = self.adversarial_loss(fake_validity, fake)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        d_loss.backward()
        self.discriminator_optimizer.step()
        
        return {
            'generator_loss': total_g_loss.item(),
            'discriminator_loss': d_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'age_classifier_loss': age_loss.item()
        }
    
    def train(self, 
              dataloader: DataLoader,
              num_epochs: int,
              save_interval: int = 10,
              checkpoint_dir: str = 'checkpoints'):
        """Train the model"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            epoch_losses = {
                'generator_loss': 0,
                'discriminator_loss': 0,
                'reconstruction_loss': 0,
                'age_classifier_loss': 0
            }
            
            for batch_idx, (real_images, age_labels) in enumerate(dataloader):
                # Generate random target ages for aging
                target_ages = torch.randint(0, self.model.num_age_groups, (real_images.size(0),))
                
                # Training step
                losses = self.train_step(real_images, age_labels, target_ages)
                
                # Accumulate losses
                for key in epoch_losses:
                    epoch_losses[key] += losses[key]
                
                # Print progress
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] '
                          f'G_Loss: {losses["generator_loss"]:.4f} '
                          f'D_Loss: {losses["discriminator_loss"]:.4f} '
                          f'Recon_Loss: {losses["reconstruction_loss"]:.4f}')
            
            # Average losses for the epoch
            for key in epoch_losses:
                epoch_losses[key] /= len(dataloader)
                self.history[key].append(epoch_losses[key])
            
            print(f'Epoch [{epoch}/{num_epochs}] Average Losses: {epoch_losses}')
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'{checkpoint_dir}/face_aging_model_epoch_{epoch+1}.pth')
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'age_classifier_optimizer_state_dict': self.age_classifier_optimizer.state_dict(),
            'history': self.history
        }, filepath)
        print(f'Checkpoint saved to {filepath}')
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.age_classifier_optimizer.load_state_dict(checkpoint['age_classifier_optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f'Checkpoint loaded from {filepath}')


def create_face_aging_model(input_size: int = 128,
                           latent_dim: int = 100,
                           age_embedding_dim: int = 64,
                           num_age_groups: int = 10) -> FaceAgingModel:
    """Factory function to create a face aging model"""
    return FaceAgingModel(
        input_size=input_size,
        latent_dim=latent_dim,
        age_embedding_dim=age_embedding_dim,
        num_age_groups=num_age_groups
    )


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_face_aging_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = FaceAgingTrainer(model, device)
    
    # Example training (uncomment to run)
    # dataset = FaceDataset('data/processed')
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    # trainer.train(dataloader, num_epochs=100) 