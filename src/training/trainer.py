"""
Training script for Face Aging Models
Handles training of CAAE, GAN, and other face aging models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
from typing import Dict, List, Optional, Tuple
import wandb
from datetime import datetime

class FaceAgingTrainer:
    """Trainer class for face aging models"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: str = 'cuda'):
        """
        Initialize the trainer
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
        # Logging and checkpointing
        self.log_dir = config.get('log_dir', 'logs')
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize wandb if enabled
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project="face-aging-model", config=config)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        optimizer_name = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0)
        
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', None)
        
        if scheduler_name is None:
            return None
        elif scheduler_name.lower() == 'step':
            step_size = self.config.get('scheduler_step_size', 30)
            gamma = self.config.get('scheduler_gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.get('epochs', 100))
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function"""
        loss_name = self.config.get('loss', 'mse')
        
        if loss_name.lower() == 'mse':
            return nn.MSELoss()
        elif loss_name.lower() == 'l1':
            return nn.L1Loss()
        elif loss_name.lower() == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_name.lower() == 'bce':
            return nn.BCELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            ages = batch['age'].to(self.device)
            age_bins = batch['age_bin'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'forward'):
                # Standard model forward pass
                outputs = self.model(images, age_bins)
                loss = self.criterion(outputs, images)  # Reconstruction loss
            else:
                # Handle different model types
                loss = self._compute_model_loss(images, ages, age_bins)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            self.optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train_batch_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def _compute_model_loss(self, images: torch.Tensor, ages: torch.Tensor, age_bins: torch.Tensor) -> torch.Tensor:
        """Compute loss for different model types"""
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(images, ages, age_bins)
        else:
            # Default reconstruction loss
            outputs = self.model(images)
            return self.criterion(outputs, images)
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                images = batch['image'].to(self.device)
                ages = batch['age'].to(self.device)
                age_bins = batch['age_bin'].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    outputs = self.model(images, age_bins)
                    loss = self.criterion(outputs, images)
                else:
                    loss = self._compute_model_loss(images, ages, age_bins)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int):
        """Train the model for specified number of epochs"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            self.train_losses.append(metrics['train_loss'])
            self.val_losses.append(metrics['val_loss'])
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Train Loss: {metrics['train_loss']:.4f}, "
                  f"Val Loss: {metrics['val_loss']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    **metrics
                })
            
            # Save checkpoint
            is_best = metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = metrics['val_loss']
            
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0 or is_best:
                self.save_checkpoint(is_best)
        
        # Final save
        self.save_checkpoint()
        
        # Training summary
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses[-50:], label='Training Loss (last 50)')
        plt.plot(self.val_losses[-50:], label='Validation Loss (last 50)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress (Zoomed)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.show()

class CAEETrainer(FaceAgingTrainer):
    """Specialized trainer for CAAE model"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: str = 'cuda'):
        super().__init__(model, train_loader, val_loader, config, device)
        
        # CAAE-specific components
        self.loss_fn = self._create_caae_loss()
        
        # Separate optimizers for different components
        self.ae_optimizer = optim.Adam([
            {'params': self.model.encoder.parameters()},
            {'params': self.model.decoder.parameters()},
            {'params': self.model.age_classifier.parameters()}
        ], lr=config.get('ae_learning_rate', 0.001))
        
        self.d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=config.get('d_learning_rate', 0.0002)
        )
    
    def _create_caae_loss(self):
        """Create CAAE loss function"""
        from src.models.caae import CAEELoss
        return CAEELoss(
            lambda_recon=self.config.get('lambda_recon', 10.0),
            lambda_age=self.config.get('lambda_age', 1.0),
            lambda_adv=self.config.get('lambda_adv', 1.0)
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train CAAE for one epoch"""
        self.model.train()
        total_ae_loss = 0.0
        total_d_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            age_bins = batch['age_bin'].to(self.device)
            
            batch_size = images.size(0)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            # Real images
            real_real, real_age = self.model.discriminate(images)
            
            # Generated images
            with torch.no_grad():
                z = self.model.encode(images)
                fake_images = self.model.decode(z, age_bins)
            
            fake_real, fake_age = self.model.discriminate(fake_images.detach())
            
            d_loss = self.loss_fn.compute_discriminator_loss(
                real_real, fake_real, real_age, fake_age, age_bins
            )
            
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Autoencoder
            self.ae_optimizer.zero_grad()
            
            # Reconstruct images
            z = self.model.encode(images)
            reconstructed = self.model.decode(z, age_bins)
            
            # Age classification
            age_predictions = self.model.classify_age(reconstructed)
            
            # Adversarial loss
            fake_real, _ = self.model.discriminate(reconstructed)
            
            ae_loss, recon_loss, age_loss = self.loss_fn.compute_autoencoder_loss(
                images, reconstructed, age_predictions, age_bins
            )
            
            adv_loss = self.loss_fn.compute_generator_loss(fake_real)
            total_ae_loss = ae_loss + self.config.get('lambda_adv', 1.0) * adv_loss
            
            total_ae_loss.backward()
            self.ae_optimizer.step()
            
            # Update progress
            progress_bar.set_postfix({
                'AE Loss': f'{total_ae_loss.item():.4f}',
                'D Loss': f'{d_loss.item():.4f}'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'ae_loss': total_ae_loss.item(),
                    'd_loss': d_loss.item(),
                    'recon_loss': recon_loss.item(),
                    'age_loss': age_loss.item(),
                    'adv_loss': adv_loss.item()
                })
        
        return {
            'train_ae_loss': total_ae_loss / num_batches,
            'train_d_loss': total_d_loss / num_batches
        }

def create_trainer(model: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  config: Dict,
                  device: str = 'cuda') -> FaceAgingTrainer:
    """Factory function to create appropriate trainer"""
    
    model_type = config.get('model_type', 'basic')
    
    if model_type.lower() == 'caae':
        return CAEETrainer(model, train_loader, val_loader, config, device)
    else:
        return FaceAgingTrainer(model, train_loader, val_loader, config, device) 