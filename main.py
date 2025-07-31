#!/usr/bin/env python3
"""
Face Aging Model - Main Application
==================================

A complete face aging application using Conditional Adversarial Autoencoders (CAAE).
This script provides a unified interface for training, inference, and web application.

Usage:
    python main.py [command] [options]

Commands:
    train       - Train the face aging model
    inference   - Run inference on images
    web         - Launch the Streamlit web application
    demo        - Run a quick demo with sample images
    setup       - Setup the project structure and dependencies
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any

def setup_project():
    """Setup the project structure and dependencies"""
    print("🚀 Setting up Face Aging Model project...")
    
    # Create directories
    directories = [
        'data/raw',
        'data/processed',
        'data/generated',
        'checkpoints',
        'logs',
        'outputs',
        'outputs/samples',
        'outputs/plots',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create sample configuration
    config = {
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
            'age_range': [0, 100],
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
            'experiment_name': 'face-aging-experiment'
        },
        'device': 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'
    }
    
    config_path = 'configs/default_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created default configuration: {config_path}")
    
    # Create sample data structure
    sample_data_structure = """
    data/
    ├── raw/                    # Put your raw face images here
    │   ├── person1/
    │   │   ├── age_25_photo1.jpg
    │   │   └── age_25_photo2.jpg
    │   └── person2/
    │       ├── age_45_photo1.jpg
    │       └── age_45_photo2.jpg
    ├── processed/              # Preprocessed images will be saved here
    │   ├── train/
    │   └── val/
    └── generated/              # Generated aged faces will be saved here
    """
    
    with open('data/README.md', 'w') as f:
        f.write("# Data Directory Structure\n\n")
        f.write("This directory contains the face images for training and testing.\n\n")
        f.write("## Structure\n")
        f.write(sample_data_structure)
        f.write("\n## Naming Convention\n")
        f.write("- Use format: `age_XX_filename.jpg` where XX is the age\n")
        f.write("- Example: `age_25_selfie.jpg`, `age_45_portrait.jpg`\n")
    
    print("✅ Created data structure documentation")
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found. Install with: pip install torch torchvision")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
    
    try:
        import mediapipe
        print(f"✅ MediaPipe {mediapipe.__version__}")
    except ImportError:
        print("❌ MediaPipe not found. Install with: pip install mediapipe")
    
    print("\n🎉 Setup complete! You can now:")
    print("1. Add face images to data/raw/")
    print("2. Run 'python main.py train' to train the model")
    print("3. Run 'python main.py web' to launch the web app")

def train_model(args):
    """Train the face aging model"""
    print("🏋️ Starting model training...")
    
    # Check if data exists
    if not os.path.exists('data/processed/train') or not os.listdir('data/processed/train'):
        print("❌ No training data found!")
        print("Please:")
        print("1. Add face images to data/raw/")
        print("2. Run preprocessing to create data/processed/")
        return
    
    # Import training module
    try:
        from src.training.train_face_aging import FaceAgingTrainingManager, create_default_config
    except ImportError as e:
        print(f"❌ Error importing training module: {e}")
        return
    
    # Create or load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Update config with command line arguments
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.resume:
        config['training']['resume_from'] = args.resume
    if args.wandb:
        config['logging']['use_wandb'] = True
    
    # Start training
    trainer = FaceAgingTrainingManager(config)
    trainer.train()

def run_inference(args):
    """Run inference on images"""
    print("🔮 Running face aging inference...")
    
    # Check if model exists
    model_path = args.model or "checkpoints/face_aging_model_best.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first with: python main.py train")
        return
    
    # Import inference module
    try:
        from src.inference.face_aging_inference import FaceAgingInference
    except ImportError as e:
        print(f"❌ Error importing inference module: {e}")
        return
    
    # Create inference pipeline
    inference = FaceAgingInference(model_path)
    
    # Process single image
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        
        print(f"Processing image: {args.image}")
        results = inference.generate_age_progression(args.image, args.ages)
        
        if results.get("success"):
            # Save results
            output_dir = args.output or "outputs/inference"
            base_filename = Path(args.image).stem
            saved_paths = inference.save_results(results, output_dir, base_filename)
            
            print(f"✅ Results saved to: {output_dir}")
            for path in saved_paths:
                print(f"  - {path}")
        else:
            print(f"❌ Error: {results.get('error', 'Unknown error')}")
    
    # Process directory
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"❌ Input directory not found: {args.input_dir}")
            return
        
        # Get image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_paths:
            print(f"❌ No images found in {args.input_dir}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Process images
        output_dir = args.output or "outputs/batch_inference"
        results = inference.batch_process(image_paths, output_dir, args.ages)
        
        print(f"\n✅ Processing complete!")
        print(f"Successfully processed: {len(results['processed'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Results saved to: {output_dir}")

def launch_web_app():
    """Launch the Streamlit web application"""
    print("🌐 Launching Face Aging Web Application...")
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        return
    
    # Launch the app
    app_path = "src/inference/streamlit_app.py"
    if not os.path.exists(app_path):
        print(f"❌ Web app not found at {app_path}")
        return
    
    print("🚀 Starting Streamlit server...")
    print("The web app will open in your browser automatically.")
    print("If it doesn't, go to: http://localhost:8501")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

def run_demo():
    """Run a quick demo with sample images"""
    print("🎬 Running Face Aging Demo...")
    
    # Check if model exists
    model_path = "checkpoints/face_aging_model_best.pth"
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        print("Please train the model first with: python main.py train")
        return
    
    # Import required modules
    try:
        from src.inference.face_aging_inference import FaceAgingInference
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        return
    
    # Create sample image if none exists
    demo_image_path = "outputs/demo_sample.jpg"
    if not os.path.exists(demo_image_path):
        print("Creating sample image for demo...")
        
        # Create a simple face-like image
        img_size = 128
        img = Image.new('RGB', (img_size, img_size), color='white')
        
        # Draw a simple face
        draw = ImageDraw.Draw(img)
        
        # Face outline
        draw.ellipse([20, 20, img_size-20, img_size-20], outline='black', width=2)
        
        # Eyes
        draw.ellipse([40, 50, 60, 70], fill='black')
        draw.ellipse([68, 50, 88, 70], fill='black')
        
        # Nose
        draw.line([64, 70, 64, 85], fill='black', width=2)
        
        # Mouth
        draw.arc([50, 80, 78, 95], start=0, end=180, fill='black', width=2)
        
        os.makedirs("outputs", exist_ok=True)
        img.save(demo_image_path)
        print(f"✅ Created demo image: {demo_image_path}")
    
    # Run inference
    print("Running inference on demo image...")
    inference = FaceAgingInference(model_path)
    
    # Generate age progression
    results = inference.generate_age_progression(demo_image_path, [0, 2, 4, 6, 8])
    
    if results.get("success"):
        # Save results
        output_dir = "outputs/demo"
        saved_paths = inference.save_results(results, output_dir, "demo")
        
        print(f"✅ Demo results saved to: {output_dir}")
        print("Generated age progressions:")
        for age_key, age_data in results["aged_faces"].items():
            print(f"  - {age_data['description']}")
        
        # Show comparison image path
        comparison_path = os.path.join(output_dir, "demo_comparison.jpg")
        if os.path.exists(comparison_path):
            print(f"📊 Comparison image: {comparison_path}")
    else:
        print(f"❌ Demo failed: {results.get('error', 'Unknown error')}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Face Aging Model - Complete AI-powered face aging application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup                    # Setup project structure
  python main.py train --epochs 50        # Train model for 50 epochs
  python main.py inference --image photo.jpg  # Process single image
  python main.py web                      # Launch web application
  python main.py demo                     # Run quick demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    subparsers.add_parser('setup', help='Setup project structure and dependencies')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the face aging model')
    train_parser.add_argument('--config', type=str, help='Path to configuration file')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size for training')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--wandb', action='store_true', help='Use WandB logging')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on images')
    inference_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    inference_parser.add_argument('--image', type=str, help='Path to single image')
    inference_parser.add_argument('--input-dir', type=str, help='Directory containing images')
    inference_parser.add_argument('--output', type=str, help='Output directory')
    inference_parser.add_argument('--ages', type=int, nargs='+', default=[0, 2, 4, 6, 8], 
                                 help='Target age groups (0-9)')
    
    # Web command
    subparsers.add_parser('web', help='Launch the Streamlit web application')
    
    # Demo command
    subparsers.add_parser('demo', help='Run a quick demo with sample images')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'setup':
        setup_project()
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'inference':
        run_inference(args)
    elif args.command == 'web':
        launch_web_app()
    elif args.command == 'demo':
        run_demo()

if __name__ == "__main__":
    main() 