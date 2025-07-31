#!/usr/bin/env python3
"""
Setup script for Face Aging Model
Installs dependencies and sets up the project
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/generated",
        "models/cnn",
        "models/gan",
        "models/caae",
        "models/age_cgan",
        "notebooks/section1_basics",
        "notebooks/section2_cnn",
        "notebooks/section3_transformers",
        "notebooks/section4_gan",
        "notebooks/section5_face_aging",
        "src/data",
        "src/models",
        "src/training",
        "src/inference",
        "utils",
        "configs",
        "logs",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def download_models():
    """Download pre-trained models if available"""
    print("🤖 Checking for pre-trained models...")
    
    # This would download pre-trained models from a repository
    # For now, we'll just create placeholder files
    model_files = [
        "checkpoints/.gitkeep",
        "models/.gitkeep"
    ]
    
    for file_path in model_files:
        Path(file_path).touch()
    
    print("✅ Model directories prepared")

def setup_git():
    """Setup git repository if not already done"""
    if not Path(".git").exists():
        print("🔧 Initializing git repository...")
        try:
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit: Face Aging Model"], check=True)
            print("✅ Git repository initialized")
        except subprocess.CalledProcessError:
            print("⚠️ Git not available, skipping git setup")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# Data
data/raw/*
data/processed/*
data/generated/*
!data/*/.gitkeep

# Models and checkpoints
checkpoints/*
!checkpoints/.gitkeep
models/*.pth
models/*.pt

# Logs
logs/*
!logs/.gitkeep

# Environment variables
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    
    print("✅ .gitignore created")

def test_installation():
    """Test if the installation was successful"""
    print("🧪 Testing installation...")
    
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import streamlit
        import face_recognition
        
        print("✅ All core dependencies imported successfully")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA not available, will use CPU")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Face Aging Model...")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install requirements
    install_requirements()
    
    # Download models
    download_models()
    
    # Setup git
    setup_git()
    
    # Create gitignore
    create_gitignore()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Generate dataset: python src/data/dataset_generator.py")
        print("2. Start training: python train.py")
        print("3. Run web app: streamlit run src/inference/app.py")
        print("4. Open notebooks: jupyter notebook")
        print("\n📚 Check README.md for detailed instructions")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 