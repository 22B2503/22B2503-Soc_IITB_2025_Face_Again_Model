"""
Conditional Adversarial Autoencoder (CAAE) for Face Aging
Implementation based on the CAAE paper for age progression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class Encoder(nn.Module):
    """Encoder network for CAAE"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 100):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)  # 128x128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 64x64
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 32x32
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # 16x16
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)  # 8x8
        
        # Batch normalization
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Fully connected layers for latent representation
        self.fc = nn.Linear(512 * 8 * 8, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        
        # Flatten and encode
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        
        return z

class Decoder(nn.Module):
    """Decoder network for CAAE"""
    
    def __init__(self, latent_dim: int = 100, output_channels: int = 3):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Fully connected layer
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        
        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)  # 16x16
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 32x32
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 64x64
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 128x128
        self.deconv5 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)  # 256x256
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Decode latent representation
        x = self.fc(z)
        x = x.view(x.size(0), 512, 8, 8)
        
        # Transposed convolutional layers
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        
        return x

class Discriminator(nn.Module):
    """Discriminator network for CAAE"""
    
    def __init__(self, input_channels: int = 3, num_age_classes: int = 3):
        super(Discriminator, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        # Batch normalization
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Fully connected layers
        self.fc_real = nn.Linear(512 * 16 * 16, 1)  # Real/fake classification
        self.fc_age = nn.Linear(512 * 16 * 16, num_age_classes)  # Age classification
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convolutional layers
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Outputs
        real_fake = torch.sigmoid(self.fc_real(x))
        age_class = self.fc_age(x)
        
        return real_fake, age_class

class AgeClassifier(nn.Module):
    """Age classifier for CAAE"""
    
    def __init__(self, input_channels: int = 3, num_age_classes: int = 3):
        super(AgeClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        # Batch normalization
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Fully connected layers
        self.fc = nn.Linear(512 * 16 * 16, num_age_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class CAAE(nn.Module):
    """Conditional Adversarial Autoencoder for Face Aging"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 latent_dim: int = 100,
                 num_age_classes: int = 3,
                 image_size: int = 256):
        super(CAAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_age_classes = num_age_classes
        self.image_size = image_size
        
        # Networks
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)
        self.discriminator = Discriminator(input_channels, num_age_classes)
        self.age_classifier = AgeClassifier(input_channels, num_age_classes)
        
        # Age embedding
        self.age_embedding = nn.Embedding(num_age_classes, latent_dim)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image to latent representation"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor, age_condition: torch.Tensor) -> torch.Tensor:
        """Decode latent representation with age conditioning"""
        # Add age condition to latent representation
        age_emb = self.age_embedding(age_condition)
        z_conditioned = z + age_emb
        
        return self.decoder(z_conditioned)
    
    def forward(self, x: torch.Tensor, age_condition: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode, condition, and decode"""
        z = self.encode(x)
        return self.decode(z, age_condition)
    
    def discriminate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discriminate between real and fake images"""
        return self.discriminator(x)
    
    def classify_age(self, x: torch.Tensor) -> torch.Tensor:
        """Classify age of input image"""
        return self.age_classifier(x)

class CAEELoss:
    """Loss functions for CAAE training"""
    
    def __init__(self, lambda_recon: float = 10.0, 
                 lambda_age: float = 1.0,
                 lambda_adv: float = 1.0):
        self.lambda_recon = lambda_recon
        self.lambda_age = lambda_age
        self.lambda_adv = lambda_adv
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()
        self.age_classification_loss = nn.CrossEntropyLoss()
        
    def compute_autoencoder_loss(self, 
                               real_images: torch.Tensor,
                               reconstructed_images: torch.Tensor,
                               age_predictions: torch.Tensor,
                               age_labels: torch.Tensor) -> torch.Tensor:
        """Compute autoencoder loss"""
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed_images, real_images)
        
        # Age classification loss
        age_loss = self.age_classification_loss(age_predictions, age_labels)
        
        total_loss = self.lambda_recon * recon_loss + self.lambda_age * age_loss
        
        return total_loss, recon_loss, age_loss
    
    def compute_discriminator_loss(self,
                                 real_discriminations: torch.Tensor,
                                 fake_discriminations: torch.Tensor,
                                 real_age_predictions: torch.Tensor,
                                 fake_age_predictions: torch.Tensor,
                                 age_labels: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss"""
        batch_size = real_discriminations.size(0)
        
        # Real/fake discrimination loss
        real_labels = torch.ones(batch_size, 1).to(real_discriminations.device)
        fake_labels = torch.zeros(batch_size, 1).to(fake_discriminations.device)
        
        real_loss = self.adversarial_loss(real_discriminations, real_labels)
        fake_loss = self.adversarial_loss(fake_discriminations, fake_labels)
        
        # Age classification loss
        real_age_loss = self.age_classification_loss(real_age_predictions, age_labels)
        fake_age_loss = self.age_classification_loss(fake_age_predictions, age_labels)
        
        total_loss = real_loss + fake_loss + self.lambda_age * (real_age_loss + fake_age_loss)
        
        return total_loss
    
    def compute_generator_loss(self, fake_discriminations: torch.Tensor) -> torch.Tensor:
        """Compute generator loss"""
        batch_size = fake_discriminations.size(0)
        real_labels = torch.ones(batch_size, 1).to(fake_discriminations.device)
        
        return self.adversarial_loss(fake_discriminations, real_labels)

def create_caae_model(config: dict) -> CAAE:
    """Factory function to create CAAE model from config"""
    return CAAE(
        input_channels=config.get('input_channels', 3),
        latent_dim=config.get('latent_dim', 100),
        num_age_classes=config.get('num_age_classes', 3),
        image_size=config.get('image_size', 256)
    )

def create_caae_loss(config: dict) -> CAEELoss:
    """Factory function to create CAAE loss from config"""
    return CAEELoss(
        lambda_recon=config.get('lambda_recon', 10.0),
        lambda_age=config.get('lambda_age', 1.0),
        lambda_adv=config.get('lambda_adv', 1.0)
    ) 