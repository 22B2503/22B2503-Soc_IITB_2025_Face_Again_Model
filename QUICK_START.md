# Quick Start Guide 🚀

Get your Face Aging Model up and running in minutes!

## Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **8GB+ RAM** (16GB+ recommended)
- **GPU** (optional but recommended for faster training)
- **Git** (for version control)

## Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd FACE_AGING_MODEL
```

### 2. Run Setup Script
```bash
python setup.py
```

This will:
- ✅ Install all dependencies
- ✅ Create project directories
- ✅ Set up git repository
- ✅ Test the installation

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print('OpenCV installed successfully')"
```

## Quick Start Options

### Option 1: Web Application (Recommended for Beginners)
```bash
# Generate some sample data first
python src/data/dataset_generator.py

# Run the web app
streamlit run src/inference/app.py
```

Open your browser to `http://localhost:8501` and start aging faces!

### Option 2: Jupyter Notebooks (Learning Path)
```bash
# Start Jupyter
jupyter notebook
```

Navigate to `notebooks/section1_basics/01_pytorch_fundamentals.ipynb` to begin your learning journey.

### Option 3: Full Training Pipeline
```bash
# Run the complete training pipeline
python train.py
```

This will:
1. 🎨 Generate synthetic face dataset
2. 🔧 Preprocess and augment data
3. 🧠 Train the CAAE model
4. 📊 Save checkpoints and logs

## Project Structure

```
FACE AGING MODEL/
├── 📁 data/                    # Dataset storage
├── 📁 models/                  # Model implementations
├── 📁 notebooks/               # Learning notebooks (8 weeks)
├── 📁 src/                     # Source code
│   ├── data/                   # Data processing
│   ├── models/                 # Model definitions
│   ├── training/               # Training scripts
│   └── inference/              # Inference pipeline
├── 📁 configs/                 # Configuration files
├── 📁 logs/                    # Training logs
├── 📁 checkpoints/             # Model checkpoints
├── 🐍 train.py                 # Main training script
├── 🐍 setup.py                 # Setup script
└── 📖 README.md                # Detailed documentation
```

## Learning Path (8 Weeks)

### Week 1: PyTorch Fundamentals
- **File**: `notebooks/section1_basics/01_pytorch_fundamentals.ipynb`
- **Goal**: Learn tensors, autograd, and basic neural networks
- **Deliverable**: Simple age prediction model

### Week 2: Convolutional Neural Networks
- **File**: `notebooks/section2_cnn/01_cnn_basics.ipynb`
- **Goal**: Understand CNNs and image processing
- **Deliverable**: CNN for face classification

### Week 3: Transformers
- **Goal**: Learn attention mechanisms and Vision Transformers
- **Deliverable**: Transformer-based face model

### Week 4: GANs
- **Goal**: Understand Generative Adversarial Networks
- **Deliverable**: Basic GAN for image generation

### Week 5: Face Aging Models
- **Goal**: Learn CAAE and Age-conditional GANs
- **Deliverable**: Face aging model prototype

### Week 6-7: Building Pipeline
- **Goal**: Create complete inference pipeline
- **Deliverable**: Working face aging application

### Week 8: Evaluation & Deployment
- **Goal**: Optimize and deploy the model
- **Deliverable**: Final polished application

## Common Commands

### Data Generation
```bash
# Generate synthetic dataset
python src/data/dataset_generator.py

# Preprocess data
python src/data/preprocessing.py
```

### Training
```bash
# Train with default config
python train.py

# Train with custom config
python train.py --config configs/my_config.yaml

# Train for specific epochs
python train.py --epochs 50

# Skip data generation (if already done)
python train.py --skip-data-gen
```

### Inference
```bash
# Run web application
streamlit run src/inference/app.py

# Run with custom port
streamlit run src/inference/app.py --server.port 8502
```

### Development
```bash
# Start Jupyter notebook
jupyter notebook

# Run tests
python -m pytest tests/

# Format code
black src/
```

## Configuration

### Training Configuration
Edit `configs/training_config.yaml` to customize:
- Model architecture
- Training parameters
- Data augmentation
- Hardware settings

### Model Types
- **CAAE**: Conditional Adversarial Autoencoder (default)
- **GAN**: Generative Adversarial Network
- **Basic**: Simple autoencoder

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config
batch_size: 8  # instead of 16
```

**2. Import Errors**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**3. Face Detection Issues**
```bash
# Install dlib dependencies
# On Ubuntu: sudo apt-get install cmake
# On Windows: Install Visual Studio Build Tools
```

**4. Slow Training**
- Use GPU if available
- Reduce image size in config
- Use mixed precision training

### Getting Help

1. 📖 Check the detailed README.md
2. 🔍 Search existing issues
3. 💬 Create a new issue with:
   - Error message
   - System information
   - Steps to reproduce

## Performance Tips

### For Training
- Use GPU (CUDA) for 10x speedup
- Increase batch size if memory allows
- Use mixed precision training
- Enable data loading optimization

### For Inference
- Use smaller model for faster inference
- Enable model quantization
- Use batch processing for multiple images

## Next Steps

1. 🎯 **Complete the 8-week learning path**
2. 🔬 **Experiment with different architectures**
3. 📊 **Try different datasets**
4. 🚀 **Deploy your model**
5. 🤝 **Contribute to the project**

## Support

- 📧 Email: your-email@example.com
- 💬 Discord: [Join our community]
- 📖 Documentation: [Full documentation]
- 🐛 Issues: [GitHub Issues]

---

**Happy coding! 🎉**

Remember: This is a learning project. Take your time, experiment, and have fun building something amazing! 