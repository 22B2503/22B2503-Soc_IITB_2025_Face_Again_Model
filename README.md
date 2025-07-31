# 🧬 Face Aging AI Model

A complete, production-ready face aging application using **Conditional Adversarial Autoencoders (CAAE)** and modern deep learning techniques. This project allows you to see how someone might look at different ages with realistic AI-generated age progressions.

## ✨ Features

- **🎯 Advanced AI Model**: CAAE with age-conditional generation
- **🔍 Face Detection & Alignment**: Using MediaPipe for robust face processing
- **🌐 Web Application**: Beautiful Streamlit interface
- **📊 Training Pipeline**: Complete training with logging and visualization
- **🎨 Multiple Age Groups**: 10 different age ranges (0-100 years)
- **💾 Batch Processing**: Process multiple images at once
- **📈 Progress Tracking**: WandB integration for experiment tracking
- **🔄 Model Checkpointing**: Resume training from any point

## 🚀 Quick Start

### 1. Setup Project
```bash
# Clone the repository
git clone <repository-url>
cd FACE_AGING_MODEL

# Install dependencies
pip install -r requirements.txt

# Setup project structure
python main.py setup
```

### 2. Add Your Data
```bash
# Add face images to data/raw/
# Use naming convention: age_XX_filename.jpg
# Example: age_25_selfie.jpg, age_45_portrait.jpg
```

### 3. Train the Model
```bash
# Train with default settings
python main.py train

# Train with custom parameters
python main.py train --epochs 100 --batch-size 32 --wandb
```

### 4. Use the Model
```bash
# Launch web application
python main.py web

# Process single image
python main.py inference --image photo.jpg

# Process multiple images
python main.py inference --input-dir photos/ --output results/

# Run demo
python main.py demo
```

## 📁 Project Structure

```
FACE_AGING_MODEL/
├── main.py                          # Main application entry point
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── data/                           # Dataset and processed data
│   ├── raw/                       # Raw face images
│   ├── processed/                 # Preprocessed images
│   └── generated/                 # Generated aged faces
├── src/                           # Source code
│   ├── models/                    # Model implementations
│   │   ├── face_aging_model.py   # Main CAAE model
│   │   └── caae.py              # Original CAAE implementation
│   ├── data/                     # Data processing
│   │   ├── preprocessing.py      # Face detection & alignment
│   │   ├── dataloader.py         # Data loading utilities
│   │   └── dataset_generator.py  # Dataset generation
│   ├── training/                 # Training scripts
│   │   └── train_face_aging.py   # Complete training pipeline
│   └── inference/                # Inference pipeline
│       ├── face_aging_inference.py # Core inference logic
│       └── streamlit_app.py      # Web application
├── checkpoints/                   # Model checkpoints
├── logs/                         # Training logs
├── outputs/                      # Generated outputs
│   ├── samples/                  # Training samples
│   └── plots/                    # Training plots
├── configs/                      # Configuration files
└── notebooks/                    # Jupyter notebooks for learning
```

## 🧠 Model Architecture

### CAAE (Conditional Adversarial Autoencoder)

The model consists of four main components:

1. **Encoder**: Compresses face images to latent representations
2. **Decoder**: Reconstructs faces with age conditioning
3. **Discriminator**: Distinguishes real from generated faces
4. **Age Classifier**: Ensures correct age progression

### Key Features:
- **Age Embeddings**: Learn age-specific features
- **Adversarial Training**: Generate realistic faces
- **Age Classification**: Ensure accurate age progression
- **Reconstruction Loss**: Maintain identity preservation

## 🎯 Usage Examples

### Command Line Interface

```bash
# Setup project
python main.py setup

# Train model
python main.py train --epochs 100 --batch-size 16

# Process single image
python main.py inference --image photo.jpg --ages 0 2 4 6 8

# Launch web app
python main.py web

# Run demo
python main.py demo
```

### Python API

```python
from src.inference.face_aging_inference import FaceAgingInference

# Load model
inference = FaceAgingInference("checkpoints/face_aging_model_best.pth")

# Generate age progression
results = inference.generate_age_progression("photo.jpg", [0, 2, 4, 6, 8])

# Save results
inference.save_results(results, "outputs/", "my_photo")
```

### Web Application

1. Run `python main.py web`
2. Open browser to `http://localhost:8501`
3. Upload image and select age groups
4. View and download results

## 📊 Training

### Data Requirements

- **Format**: JPG, PNG images
- **Naming**: `age_XX_filename.jpg` (XX = age)
- **Quality**: Clear, front-facing portraits
- **Quantity**: 1000+ images recommended

### Training Configuration

```json
{
  "model": {
    "input_size": 128,
    "latent_dim": 100,
    "age_embedding_dim": 64,
    "num_age_groups": 10
  },
  "training": {
    "num_epochs": 100,
    "batch_size": 16,
    "save_interval": 10
  }
}
```

### Training Commands

```bash
# Basic training
python main.py train

# Custom training
python main.py train --epochs 200 --batch-size 32 --wandb

# Resume training
python main.py train --resume checkpoints/face_aging_model_epoch_50.pth
```

## 🔧 Configuration

### Model Parameters

- **Input Size**: 128x128 pixels (configurable)
- **Latent Dimension**: 100 (compressed representation)
- **Age Embedding**: 64 dimensions
- **Age Groups**: 10 groups (0-10, 11-20, ..., 91-100)

### Training Parameters

- **Learning Rate**: 0.0002 (Adam optimizer)
- **Batch Size**: 16 (adjustable)
- **Loss Weights**: 
  - Adversarial: 1.0
  - Reconstruction: 10.0
  - Age Classification: 1.0

## 📈 Performance

### Expected Results

- **Training Time**: 2-4 hours on GPU
- **Inference Speed**: <5 seconds per image
- **Quality**: Realistic age progression
- **Accuracy**: 85%+ age classification accuracy

### Hardware Requirements

- **Training**: GPU with 8GB+ VRAM recommended
- **Inference**: CPU or GPU
- **Memory**: 16GB+ RAM
- **Storage**: 10GB+ free space

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- Git

### Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black flake8 pytest
```

### Verify Installation

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test imports
python -c "from src.models.face_aging_model import create_face_aging_model; print('✅ All imports successful')"
```

## 🎨 Web Application

### Features

- **📸 Image Upload**: Drag & drop or file picker
- **🎂 Age Selection**: Choose target age groups
- **🖼️ Comparison View**: Side-by-side age progression
- **📥 Download Results**: Save individual or comparison images
- **⚙️ Model Info**: View model statistics and parameters

### Launch

```bash
python main.py web
```

Then open `http://localhost:8501` in your browser.

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python main.py train --batch-size 8
   ```

2. **No Face Detected**
   - Ensure image contains clear, front-facing face
   - Check image quality and lighting
   - Try different face angles

3. **Poor Quality Results**
   - Increase training epochs
   - Use more diverse training data
   - Adjust loss weights

4. **Model Not Loading**
   ```bash
   # Check checkpoint exists
   ls checkpoints/
   
   # Train model first
   python main.py train
   ```

### Debug Mode

```bash
# Enable debug logging
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

## 📚 Learning Resources

### Theory
- [CAAE Paper](https://arxiv.org/abs/1702.01983)
- [GAN Basics](https://arxiv.org/abs/1406.2661)
- [Face Aging Survey](https://arxiv.org/abs/2006.10991)

### Implementation
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Face detection and alignment
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **Albumentations**: Image augmentation
- **WandB**: Experiment tracking

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@example.com

---

**🎉 Ready to see the future? Start aging faces with AI!** 