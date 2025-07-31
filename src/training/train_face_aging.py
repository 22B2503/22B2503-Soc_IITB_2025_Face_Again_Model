import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from datetime import datetime

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.face_aging_model import FaceAgingModel, FaceDataset, FaceAgingTrainer, create_face_aging_model
from data.preprocessing import FacePreprocessor, AgeGroupMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FaceAgingTrainingManager:
    """Complete training manager for face aging model"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self.model = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'generator_losses': [],
            'discriminator_losses': [],
            'reconstruction_losses': [],
            'age_classifier_losses': []
        }
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            'checkpoints',
            'logs',
            'outputs',
            'outputs/samples',
            'outputs/plots'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_model(self):
        """Setup the face aging model"""
        logger.info("Setting up face aging model...")
        
        self.model = create_face_aging_model(
            input_size=self.config['model']['input_size'],
            latent_dim=self.config['model']['latent_dim'],
            age_embedding_dim=self.config['model']['age_embedding_dim'],
            num_age_groups=self.config['model']['num_age_groups']
        )
        
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Create trainer
        self.trainer = FaceAgingTrainer(self.model, self.device)
        
        logger.info("Model setup complete")
    
    def setup_data(self):
        """Setup data loaders"""
        logger.info("Setting up data loaders...")
        
        # Create preprocessor
        preprocessor = FacePreprocessor(
            target_size=self.config['data']['image_size'],
            use_face_detection=self.config['data']['use_face_detection'],
            use_face_alignment=self.config['data']['use_face_alignment']
        )
        
        # Create datasets
        train_dataset = FaceDataset(
            data_dir=self.config['data']['train_dir'],
            transform=preprocessor.validation_transform,
            age_range=self.config['data']['age_range'],
            image_size=self.config['data']['image_size']
        )
        
        val_dataset = FaceDataset(
            data_dir=self.config['data']['val_dir'],
            transform=preprocessor.validation_transform,
            age_range=self.config['data']['age_range'],
            image_size=self.config['data']['image_size']
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
        logger.info("Data setup complete")
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        if self.config['logging']['use_wandb']:
            try:
                wandb.init(
                    project=self.config['logging']['wandb_project'],
                    config=self.config,
                    name=self.config['logging']['experiment_name']
                )
                logger.info("WandB logging initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.config['logging']['use_wandb'] = False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'generator_loss': 0.0,
            'discriminator_loss': 0.0,
            'reconstruction_loss': 0.0,
            'age_classifier_loss': 0.0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (real_images, age_labels) in enumerate(progress_bar):
            # Generate random target ages for aging
            target_ages = torch.randint(0, self.model.num_age_groups, (real_images.size(0),))
            
            # Training step
            losses = self.trainer.train_step(real_images, age_labels, target_ages)
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f"{losses['generator_loss']:.4f}",
                'D_Loss': f"{losses['discriminator_loss']:.4f}",
                'Recon_Loss': f"{losses['reconstruction_loss']:.4f}"
            })
            
            # Log to WandB
            if self.config['logging']['use_wandb'] and batch_idx % 100 == 0:
                wandb.log({
                    'batch_generator_loss': losses['generator_loss'],
                    'batch_discriminator_loss': losses['discriminator_loss'],
                    'batch_reconstruction_loss': losses['reconstruction_loss'],
                    'batch_age_classifier_loss': losses['age_classifier_loss'],
                    'batch': batch_idx + epoch * len(self.train_loader)
                })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_losses = {
            'generator_loss': 0.0,
            'discriminator_loss': 0.0,
            'reconstruction_loss': 0.0,
            'age_classifier_loss': 0.0
        }
        
        with torch.no_grad():
            for real_images, age_labels in tqdm(self.val_loader, desc=f"Validation {epoch}"):
                # Generate random target ages
                target_ages = torch.randint(0, self.model.num_age_groups, (real_images.size(0),))
                
                # Move to device
                real_images = real_images.to(self.device)
                age_labels = age_labels.to(self.device)
                target_ages = target_ages.to(self.device)
                
                # Forward pass
                z = self.model.encode(real_images)
                fake_images = self.model.decode(z, target_ages)
                
                # Calculate losses
                fake_validity = self.model.discriminator(fake_images)
                age_pred = self.model.age_classifier(fake_images)
                
                # Losses
                g_loss = self.trainer.adversarial_loss(fake_validity, torch.ones_like(fake_validity))
                recon_loss = self.trainer.reconstruction_loss(fake_images, real_images)
                age_loss = self.trainer.age_classification_loss(age_pred, target_ages)
                
                # Accumulate
                val_losses['generator_loss'] += g_loss.item()
                val_losses['reconstruction_loss'] += recon_loss.item()
                val_losses['age_classifier_loss'] += age_loss.item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        return val_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'encoder_optimizer_state_dict': self.trainer.encoder_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.trainer.discriminator_optimizer.state_dict(),
            'age_classifier_optimizer_state_dict': self.trainer.age_classifier_optimizer.state_dict(),
            'training_history': self.training_history,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = f"checkpoints/face_aging_model_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = "checkpoints/face_aging_model_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer states
        self.trainer.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        self.trainer.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.trainer.age_classifier_optimizer.load_state_dict(checkpoint['age_classifier_optimizer_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint['training_history']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Checkpoint loaded. Resuming from epoch {self.current_epoch}")
    
    def generate_samples(self, epoch: int, num_samples: int = 8):
        """Generate sample images for visualization"""
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch of validation images
            val_batch = next(iter(self.val_loader))
            real_images, age_labels = val_batch
            real_images = real_images[:num_samples].to(self.device)
            
            # Generate aged versions
            samples = []
            for age_group in range(self.model.num_age_groups):
                z = self.model.encode(real_images)
                aged_images = self.model.decode(z, torch.tensor([age_group] * num_samples).to(self.device))
                samples.append(aged_images)
            
            # Save samples
            self._save_sample_grid(real_images, samples, epoch)
    
    def _save_sample_grid(self, original_images: torch.Tensor, aged_samples: list, epoch: int):
        """Save a grid of sample images"""
        import matplotlib.pyplot as plt
        
        # Convert tensors to images
        def tensor_to_image(tensor):
            # Denormalize
            tensor = tensor * 0.5 + 0.5
            # Convert to numpy
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            image_array = tensor.permute(1, 2, 0).cpu().numpy()
            image_array = np.clip(image_array, 0, 1)
            return image_array
        
        # Create grid
        num_samples = original_images.size(0)
        num_ages = len(aged_samples)
        
        fig, axes = plt.subplots(num_samples, num_ages + 1, figsize=(15, 3 * num_samples))
        
        # Original images
        for i in range(num_samples):
            axes[i, 0].imshow(tensor_to_image(original_images[i]))
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
        
        # Aged images
        age_descriptions = [
            "Child", "Teen", "Young Adult", "Adult", "Middle Age",
            "Senior", "Elderly", "Senior Citizen", "Elder", "Centenarian"
        ]
        
        for age_idx, (aged_batch, age_desc) in enumerate(zip(aged_samples, age_descriptions)):
            for sample_idx in range(num_samples):
                axes[sample_idx, age_idx + 1].imshow(tensor_to_image(aged_batch[sample_idx]))
                axes[sample_idx, age_idx + 1].set_title(age_desc)
                axes[sample_idx, age_idx + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"outputs/samples/epoch_{epoch}_samples.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sample images saved for epoch {epoch}")
    
    def plot_training_history(self):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Generator loss
        axes[0, 0].plot(self.training_history['generator_losses'])
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Discriminator loss
        axes[0, 1].plot(self.training_history['discriminator_losses'])
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        
        # Reconstruction loss
        axes[1, 0].plot(self.training_history['reconstruction_losses'])
        axes[1, 0].set_title('Reconstruction Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        
        # Age classifier loss
        axes[1, 1].plot(self.training_history['age_classifier_losses'])
        axes[1, 1].set_title('Age Classifier Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('outputs/plots/training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Training history plot saved")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_wandb()
        
        # Load checkpoint if resuming
        if self.config['training']['resume_from']:
            self.load_checkpoint(self.config['training']['resume_from'])
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            logger.info(f"Starting epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            # Train
            train_losses = self.train_epoch(epoch + 1)
            
            # Validate
            val_losses = self.validate_epoch(epoch + 1)
            
            # Update history
            self.training_history['generator_losses'].append(train_losses['generator_loss'])
            self.training_history['discriminator_losses'].append(train_losses['discriminator_loss'])
            self.training_history['reconstruction_losses'].append(train_losses['reconstruction_loss'])
            self.training_history['age_classifier_losses'].append(train_losses['age_classifier_loss'])
            
            # Log losses
            total_train_loss = sum(train_losses.values())
            total_val_loss = sum(val_losses.values())
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}")
            
            # Log to WandB
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_generator_loss': train_losses['generator_loss'],
                    'train_discriminator_loss': train_losses['discriminator_loss'],
                    'train_reconstruction_loss': train_losses['reconstruction_loss'],
                    'train_age_classifier_loss': train_losses['age_classifier_loss'],
                    'val_generator_loss': val_losses['generator_loss'],
                    'val_discriminator_loss': val_losses['discriminator_loss'],
                    'val_reconstruction_loss': val_losses['reconstruction_loss'],
                    'val_age_classifier_loss': val_losses['age_classifier_loss'],
                    'total_train_loss': total_train_loss,
                    'total_val_loss': total_val_loss
                })
            
            # Save checkpoint
            is_best = total_val_loss < self.best_loss
            if is_best:
                self.best_loss = total_val_loss
            
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch + 1, is_best)
            
            # Generate samples
            if (epoch + 1) % self.config['training']['sample_interval'] == 0:
                self.generate_samples(epoch + 1)
        
        # Final save and plots
        self.save_checkpoint(self.config['training']['num_epochs'])
        self.plot_training_history()
        
        logger.info("Training completed!")
        
        # Close WandB
        if self.config['logging']['use_wandb']:
            wandb.finish()


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration"""
    return {
        'model': {
            'input_size': 128,
            'latent_dim': 100,
            'age_embedding_dim': 64,
            'num_age_groups': 10
        },
        'data': {
            'train_dir': 'data/processed/train',
            'val_dir': 'data/processed/val',
            'image_size': 128,
            'age_range': (0, 100),
            'use_face_detection': True,
            'use_face_alignment': True
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 16,
            'num_workers': 4,
            'save_interval': 10,
            'sample_interval': 5,
            'resume_from': None
        },
        'logging': {
            'use_wandb': False,
            'wandb_project': 'face-aging-model',
            'experiment_name': f'face-aging-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Face Aging Model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--wandb', action='store_true', help='Use WandB logging')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Update config with command line arguments
    if args.resume:
        config['training']['resume_from'] = args.resume
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.wandb:
        config['logging']['use_wandb'] = True
    
    # Save config
    os.makedirs('configs', exist_ok=True)
    config_path = f"configs/training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")
    
    # Start training
    trainer = FaceAgingTrainingManager(config)
    trainer.train()


if __name__ == "__main__":
    main() 