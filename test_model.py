#!/usr/bin/env python3
"""
Test Script for Face Aging Model
================================

This script tests all components of the face aging model to ensure everything works correctly.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from models.face_aging_model import FaceAgingModel, create_face_aging_model
        print("✅ Face aging model imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import face aging model: {e}")
        return False
    
    try:
        from data.preprocessing import FacePreprocessor, AgeGroupMapper
        print("✅ Preprocessing modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import preprocessing modules: {e}")
        return False
    
    try:
        from inference.face_aging_inference import FaceAgingInference
        print("✅ Inference module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import inference module: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation and forward pass"""
    print("\n🧠 Testing model creation...")
    
    try:
        from models.face_aging_model import create_face_aging_model
        
        # Create model
        model = create_face_aging_model(
            input_size=128,
            latent_dim=100,
            age_embedding_dim=64,
            num_age_groups=10
        )
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 3, 128, 128)
        age_labels = torch.randint(0, 10, (batch_size,))
        
        # Test encode/decode
        z = model.encode(x)
        reconstructed = model.decode(z, age_labels)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Latent shape: {z.shape}")
        print(f"   Output shape: {reconstructed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing pipeline"""
    print("\n🔧 Testing preprocessing...")
    
    try:
        from data.preprocessing import FacePreprocessor, AgeGroupMapper
        
        # Create preprocessor
        preprocessor = FacePreprocessor(
            target_size=128,
            use_face_detection=False,  # Disable for testing
            use_face_alignment=False
        )
        
        print("✅ Preprocessor created successfully")
        
        # Test age mapper
        age_mapper = AgeGroupMapper(num_groups=10)
        age_group = age_mapper.age_to_group(25)
        age_range = age_mapper.group_to_age_range(age_group)
        
        print(f"✅ Age mapping: 25 years -> Group {age_group} ({age_range[0]}-{age_range[1]})")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_inference_pipeline():
    """Test inference pipeline with dummy model"""
    print("\n🔮 Testing inference pipeline...")
    
    try:
        from inference.face_aging_inference import FaceAgingInference
        from models.face_aging_model import create_face_aging_model
        
        # Create a dummy model checkpoint
        model = create_face_aging_model()
        
        # Create temporary checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': {
                    'model': {
                        'input_size': 128,
                        'latent_dim': 100,
                        'age_embedding_dim': 64,
                        'num_age_groups': 10
                    }
                }
            }
            torch.save(checkpoint, tmp_file.name)
            checkpoint_path = tmp_file.name
        
        # Test inference pipeline
        inference = FaceAgingInference(checkpoint_path, device='cpu')
        
        print("✅ Inference pipeline created successfully")
        
        # Clean up
        os.unlink(checkpoint_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def test_data_generation():
    """Test data generation and loading"""
    print("\n📊 Testing data generation...")
    
    try:
        from models.face_aging_model import FaceDataset
        
        # Create a temporary directory with sample images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample images
            for i in range(5):
                img = Image.new('RGB', (128, 128), color='white')
                draw = ImageDraw.Draw(img)
                
                # Draw a simple face
                draw.ellipse([20, 20, 108, 108], outline='black', width=2)
                draw.ellipse([40, 50, 60, 70], fill='black')  # Left eye
                draw.ellipse([68, 50, 88, 70], fill='black')  # Right eye
                draw.line([64, 70, 64, 85], fill='black', width=2)  # Nose
                draw.arc([50, 80, 78, 95], start=0, end=180, fill='black', width=2)  # Mouth
                
                # Save with age in filename
                age = 20 + i * 10
                img.save(os.path.join(temp_dir, f"age_{age}_test_{i}.jpg"))
            
            # Create dataset
            dataset = FaceDataset(temp_dir, image_size=128)
            
            print(f"✅ Dataset created with {len(dataset)} samples")
            
            # Test data loading
            if len(dataset) > 0:
                image, age_group = dataset[0]
                print(f"✅ Sample loaded: image shape {image.shape}, age group {age_group}")
            
            return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_cuda_availability():
    """Test CUDA availability"""
    print("\n🚀 Testing CUDA availability...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, will use CPU")
    
    return True

def create_sample_data():
    """Create sample data for testing"""
    print("\n📁 Creating sample data...")
    
    try:
        # Create data directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Create sample images
        for i in range(10):
            img = Image.new('RGB', (256, 256), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw a more detailed face
            # Face outline
            draw.ellipse([50, 50, 206, 206], outline='black', width=3)
            
            # Eyes
            draw.ellipse([80, 100, 110, 130], fill='black')
            draw.ellipse([146, 100, 176, 130], fill='black')
            
            # Nose
            draw.line([128, 130, 128, 160], fill='black', width=3)
            
            # Mouth
            draw.arc([100, 170, 156, 190], start=0, end=180, fill='black', width=3)
            
            # Hair
            draw.arc([30, 30, 226, 100], start=0, end=180, fill='brown', width=5)
            
            # Save with age in filename
            age = 20 + i * 8
            filename = f"age_{age}_sample_{i}.jpg"
            img.save(os.path.join('data/raw', filename))
        
        print(f"✅ Created 10 sample images in data/raw/")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def main():
    """Run all tests"""
    print("🧬 Face Aging Model - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Creation", test_model_creation),
        ("Preprocessing", test_preprocessing),
        ("Inference Pipeline", test_inference_pipeline),
        ("Data Generation", test_data_generation),
        ("CUDA Availability", test_cuda_availability),
        ("Sample Data Creation", create_sample_data)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The face aging model is ready to use.")
        print("\nNext steps:")
        print("1. Add real face images to data/raw/")
        print("2. Run: python main.py train")
        print("3. Run: python main.py web")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 