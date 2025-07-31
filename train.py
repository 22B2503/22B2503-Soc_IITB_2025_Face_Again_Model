#!/usr/bin/env python3
"""
Main Training Script for Face Aging Model
Orchestrates the entire training pipeline from data generation to model training
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.data.dataset_generator import FaceDatasetGenerator
from src.data.preprocessing import FacePreprocessor
from src.data.dataloader import create_data_loaders
from src.models.caae import create_caae_model
from src.training.trainer import create_trainer

def setup_environment(config):
    """Setup training environment"""
    print("🚀 Setting up training environment...")
    
    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup device
    device_config = config.get('hardware', {})
    if device_config.get('device', 'auto') == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config['device'])
    
    print(f"✅ Using device: {device}")
    
    # Create directories
    dirs_to_create = [
        'data/raw', 'data/processed', 'data/generated',
        'logs', 'checkpoints', 'models'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return device

def generate_dataset(config):
    """Generate synthetic dataset if needed"""
    print("\n📊 Generating dataset...")
    
    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir', 'data/processed')
    
    # Check if dataset already exists
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print("✅ Dataset already exists, skipping generation")
        return
    
    # Generate synthetic dataset
    generator = FaceDatasetGenerator(output_dir='data/raw')
    
    # Generate faces
    num_faces = config.get('dataset_size', 1000)
    print(f"🎨 Generating {num_faces} synthetic faces...")
    
    synthetic_files = generator.generate_synthetic_faces(num_faces)
    
    # Create metadata
    metadata_path = generator.create_metadata(synthetic_files)
    print(f"📝 Metadata saved to: {metadata_path}")
    
    # Visualize dataset
    generator.visualize_dataset(synthetic_files)
    
    print(f"✅ Generated {sum(len(files) for files in synthetic_files.values())} faces")

def preprocess_dataset(config):
    """Preprocess the dataset"""
    print("\n🔧 Preprocessing dataset...")
    
    data_config = config.get('data', {})
    input_dir = 'data/raw'
    output_dir = data_config.get('data_dir', 'data/processed')
    
    # Check if preprocessing is needed
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print("✅ Preprocessed dataset already exists, skipping preprocessing")
        return
    
    # Initialize preprocessor
    preprocessor = FacePreprocessor(
        target_size=tuple(data_config.get('target_size', [256, 256])),
        face_detection_method='face_recognition'
    )
    
    # Create augmented dataset
    augmentations_per_image = config.get('augmentations_per_image', 3)
    print(f"🔄 Creating augmented dataset with {augmentations_per_image} augmentations per image...")
    
    stats = preprocessor.create_augmented_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        augmentations_per_image=augmentations_per_image
    )
    
    print(f"✅ Preprocessing completed: {stats['total_processed']} images processed")
    
    # Create metadata
    metadata = preprocessor.create_dataset_metadata(
        processed_dir=output_dir,
        output_path=os.path.join(output_dir, 'dataset_metadata.json')
    )
    
    print(f"📝 Dataset metadata created with {metadata['dataset_info']['total_images']} images")

def create_model(config, device):
    """Create and initialize the model"""
    print("\n🧠 Creating model...")
    
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'caae')
    
    if model_type.lower() == 'caae':
        model = create_caae_model(model_config)
        print("✅ CAAE model created")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move to device
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model

def train_model(model, config, device):
    """Train the model"""
    print("\n🎯 Starting model training...")
    
    # Create data loaders
    data_config = config.get('data', {})
    train_loader, val_loader, test_loader = create_data_loaders(data_config)
    
    print(f"📊 Data loaders created:")
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    
    # Create trainer
    training_config = config.get('training', {})
    logging_config = config.get('logging', {})
    
    # Combine configs for trainer
    trainer_config = {**training_config, **logging_config}
    trainer_config['model_type'] = config.get('model', {}).get('type', 'caae')
    
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device
    )
    
    # Start training
    epochs = training_config.get('epochs', 100)
    trainer.train(epochs)
    
    print("✅ Training completed!")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train Face Aging Model')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-data-gen', action='store_true',
                       help='Skip dataset generation')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of training epochs')
    
    args = parser.parse_args()
    
    # Load configuration
    print("📋 Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs if specified
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    print(f"✅ Configuration loaded from {args.config}")
    
    # Setup environment
    device = setup_environment(config)
    
    # Generate dataset
    if not args.skip_data_gen:
        generate_dataset(config)
    
    # Preprocess dataset
    if not args.skip_preprocessing:
        preprocess_dataset(config)
    
    # Create model
    model = create_model(config, device)
    
    # Train model
    train_model(model, config, device)
    
    print("\n🎉 Training pipeline completed successfully!")
    print("📁 Check the following directories for results:")
    print("   - logs/ : Training logs and plots")
    print("   - checkpoints/ : Model checkpoints")
    print("   - data/generated/ : Generated aged faces")

if __name__ == "__main__":
    main() 