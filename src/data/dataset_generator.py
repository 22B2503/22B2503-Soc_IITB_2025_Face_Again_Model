"""
Dataset Generator for Face Aging Model
Generates synthetic face datasets using various techniques
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import json
from typing import List, Tuple, Dict, Optional
import requests
from io import BytesIO
import face_recognition
import dlib
from tqdm import tqdm
import matplotlib.pyplot as plt

class FaceDatasetGenerator:
    """Generates synthetic face datasets for face aging model training"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = None
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/young", exist_ok=True)
        os.makedirs(f"{output_dir}/middle", exist_ok=True)
        os.makedirs(f"{output_dir}/elderly", exist_ok=True)
        
    def generate_synthetic_faces(self, num_faces: int = 1000) -> Dict[str, List[str]]:
        """
        Generate synthetic faces using geometric shapes and textures
        
        Args:
            num_faces: Number of faces to generate
            
        Returns:
            Dictionary with age categories and file paths
        """
        print(f"Generating {num_faces} synthetic faces...")
        
        generated_files = {
            'young': [],
            'middle': [],
            'elderly': []
        }
        
        for i in tqdm(range(num_faces), desc="Generating faces"):
            # Generate base face
            face_img = self._create_base_face()
            
            # Apply age-specific modifications
            age_category = random.choice(['young', 'middle', 'elderly'])
            aged_face = self._apply_age_effects(face_img, age_category)
            
            # Save the face
            filename = f"face_{i:04d}_{age_category}.jpg"
            filepath = os.path.join(self.output_dir, age_category, filename)
            aged_face.save(filepath)
            
            generated_files[age_category].append(filepath)
            
        return generated_files
    
    def _create_base_face(self) -> Image.Image:
        """Create a basic synthetic face using geometric shapes"""
        # Create a 256x256 image with skin tone background
        img = Image.new('RGB', (256, 256), color=(255, 220, 177))
        draw = ImageDraw.Draw(img)
        
        # Face outline (oval)
        face_bbox = [30, 50, 226, 206]
        draw.ellipse(face_bbox, fill=(255, 220, 177), outline=(200, 150, 100), width=2)
        
        # Eyes
        left_eye = [80, 100, 110, 120]
        right_eye = [146, 100, 176, 120]
        draw.ellipse(left_eye, fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        draw.ellipse(right_eye, fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        
        # Pupils
        draw.ellipse([85, 105, 95, 115], fill=(0, 0, 0))
        draw.ellipse([151, 105, 161, 115], fill=(0, 0, 0))
        
        # Nose
        nose_points = [(128, 120), (118, 140), (138, 140)]
        draw.polygon(nose_points, fill=(240, 200, 160), outline=(200, 150, 100))
        
        # Mouth
        mouth_bbox = [100, 150, 156, 170]
        draw.ellipse(mouth_bbox, fill=(255, 200, 200), outline=(200, 100, 100), width=1)
        
        # Add some random features
        self._add_random_features(draw)
        
        return img
    
    def _add_random_features(self, draw: ImageDraw.Draw):
        """Add random facial features for variety"""
        # Eyebrows
        if random.random() > 0.5:
            # Left eyebrow
            for i in range(5):
                x = 85 + i * 5
                y = 90 + random.randint(-2, 2)
                draw.line([(x, y), (x+3, y-2)], fill=(100, 50, 0), width=2)
            
            # Right eyebrow
            for i in range(5):
                x = 151 + i * 5
                y = 90 + random.randint(-2, 2)
                draw.line([(x, y), (x+3, y-2)], fill=(100, 50, 0), width=2)
        
        # Freckles
        if random.random() > 0.7:
            for _ in range(random.randint(3, 8)):
                x = random.randint(60, 196)
                y = random.randint(80, 140)
                draw.ellipse([x, y, x+2, y+2], fill=(200, 150, 100))
    
    def _apply_age_effects(self, face_img: Image.Image, age_category: str) -> Image.Image:
        """Apply age-specific effects to the face"""
        img_array = np.array(face_img)
        
        if age_category == 'young':
            # Young faces: smooth, bright, minimal wrinkles
            img_array = self._apply_smoothing(img_array, factor=0.3)
            img_array = self._adjust_brightness(img_array, factor=1.1)
            
        elif age_category == 'middle':
            # Middle-aged: some wrinkles, slight texture changes
            img_array = self._add_wrinkles(img_array, intensity=0.3)
            img_array = self._adjust_contrast(img_array, factor=1.05)
            
        elif age_category == 'elderly':
            # Elderly: pronounced wrinkles, texture changes, gray hair
            img_array = self._add_wrinkles(img_array, intensity=0.8)
            img_array = self._add_aging_texture(img_array)
            img_array = self._adjust_brightness(img_array, factor=0.9)
            img_array = self._add_gray_hair(img_array)
        
        return Image.fromarray(img_array)
    
    def _apply_smoothing(self, img_array: np.ndarray, factor: float) -> np.ndarray:
        """Apply Gaussian smoothing to the image"""
        kernel_size = int(5 * factor)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
    
    def _adjust_brightness(self, img_array: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness"""
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def _adjust_contrast(self, img_array: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast"""
        return np.clip(img_array * factor, 0, 255).astype(np.uint8)
    
    def _add_wrinkles(self, img_array: np.ndarray, intensity: float) -> np.ndarray:
        """Add synthetic wrinkles to the face"""
        height, width = img_array.shape[:2]
        
        # Create wrinkle patterns
        wrinkles = np.zeros_like(img_array)
        
        # Forehead wrinkles
        for i in range(int(3 * intensity)):
            y = 60 + i * 15
            for x in range(80, 176):
                if random.random() < 0.3:
                    wrinkles[y, x] = [50, 30, 20]
        
        # Eye wrinkles
        for i in range(int(2 * intensity)):
            # Left eye
            center_x, center_y = 95, 110
            for angle in range(0, 360, 10):
                rad = np.radians(angle)
                x = int(center_x + 20 * np.cos(rad))
                y = int(center_y + 20 * np.sin(rad))
                if 0 <= x < width and 0 <= y < height and random.random() < 0.2:
                    wrinkles[y, x] = [40, 25, 15]
            
            # Right eye
            center_x, center_y = 161, 110
            for angle in range(0, 360, 10):
                rad = np.radians(angle)
                x = int(center_x + 20 * np.cos(rad))
                y = int(center_y + 20 * np.sin(rad))
                if 0 <= x < width and 0 <= y < height and random.random() < 0.2:
                    wrinkles[y, x] = [40, 25, 15]
        
        # Mouth wrinkles
        for i in range(int(2 * intensity)):
            y = 180 + i * 5
            for x in range(110, 146):
                if random.random() < 0.4:
                    wrinkles[y, x] = [45, 28, 18]
        
        return np.clip(img_array - wrinkles * intensity, 0, 255).astype(np.uint8)
    
    def _add_aging_texture(self, img_array: np.ndarray) -> np.ndarray:
        """Add aging texture to the skin"""
        noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
        return np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    def _add_gray_hair(self, img_array: np.ndarray) -> np.ndarray:
        """Add gray hair effect to the top of the face"""
        # Add gray hair at the top
        for y in range(20, 50):
            for x in range(60, 196):
                if random.random() < 0.3:
                    gray_value = random.randint(150, 200)
                    img_array[y, x] = [gray_value, gray_value, gray_value]
        
        return img_array
    
    def download_real_faces(self, num_faces: int = 500) -> Dict[str, List[str]]:
        """
        Download real face images from public APIs (for educational purposes)
        Note: This is a simplified version. In practice, you'd use proper datasets.
        """
        print(f"Downloading {num_faces} real face images...")
        
        # This is a placeholder for real face download
        # In practice, you would use datasets like:
        # - CelebA
        # - UTKFace
        # - IMDB-WIKI
        # - AgeDB
        
        # For now, we'll generate more synthetic faces
        return self.generate_synthetic_faces(num_faces)
    
    def create_age_progression_dataset(self, base_faces: List[str], target_ages: List[int]) -> Dict[str, List[str]]:
        """
        Create age progression dataset from base faces
        
        Args:
            base_faces: List of paths to base face images
            target_ages: List of target ages to generate
            
        Returns:
            Dictionary with age categories and file paths
        """
        print("Creating age progression dataset...")
        
        progression_files = {}
        
        for face_path in tqdm(base_faces, desc="Processing faces"):
            try:
                face_img = Image.open(face_path)
                
                for target_age in target_ages:
                    # Determine age category
                    if target_age < 30:
                        category = 'young'
                    elif target_age < 60:
                        category = 'middle'
                    else:
                        category = 'elderly'
                    
                    # Apply age progression
                    aged_face = self._apply_age_progression(face_img, target_age)
                    
                    # Save the aged face
                    base_name = os.path.splitext(os.path.basename(face_path))[0]
                    filename = f"{base_name}_age_{target_age}.jpg"
                    filepath = os.path.join(self.output_dir, category, filename)
                    aged_face.save(filepath)
                    
                    if category not in progression_files:
                        progression_files[category] = []
                    progression_files[category].append(filepath)
                    
            except Exception as e:
                print(f"Error processing {face_path}: {e}")
                continue
        
        return progression_files
    
    def _apply_age_progression(self, face_img: Image.Image, target_age: int) -> Image.Image:
        """Apply age progression to a face image"""
        img_array = np.array(face_img)
        
        # Calculate aging intensity based on target age
        if target_age < 30:
            intensity = 0.1
        elif target_age < 50:
            intensity = 0.4
        elif target_age < 70:
            intensity = 0.7
        else:
            intensity = 1.0
        
        # Apply aging effects
        img_array = self._add_wrinkles(img_array, intensity)
        img_array = self._adjust_brightness(img_array, 1.0 - intensity * 0.2)
        
        if target_age > 60:
            img_array = self._add_aging_texture(img_array)
            img_array = self._add_gray_hair(img_array)
        
        return Image.fromarray(img_array)
    
    def create_metadata(self, dataset_files: Dict[str, List[str]]) -> str:
        """Create metadata file for the dataset"""
        metadata = {
            'total_images': sum(len(files) for files in dataset_files.values()),
            'categories': {},
            'generation_info': {
                'method': 'synthetic_generation',
                'date': str(np.datetime64('now')),
                'version': '1.0'
            }
        }
        
        for category, files in dataset_files.items():
            metadata['categories'][category] = {
                'count': len(files),
                'files': files
            }
        
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def visualize_dataset(self, dataset_files: Dict[str, List[str]], num_samples: int = 5):
        """Visualize samples from the dataset"""
        fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
        
        for i, (category, files) in enumerate(dataset_files.items()):
            sample_files = random.sample(files, min(num_samples, len(files)))
            
            for j, file_path in enumerate(sample_files):
                try:
                    img = Image.open(file_path)
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f'{category.capitalize()}')
                    axes[i, j].axis('off')
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_samples.png'))
        plt.show()

def main():
    """Main function to generate the dataset"""
    generator = FaceDatasetGenerator()
    
    # Generate synthetic dataset
    print("=== Generating Synthetic Face Dataset ===")
    synthetic_files = generator.generate_synthetic_faces(num_faces=1000)
    
    # Create metadata
    metadata_path = generator.create_metadata(synthetic_files)
    print(f"Metadata saved to: {metadata_path}")
    
    # Visualize dataset
    generator.visualize_dataset(synthetic_files)
    
    print("Dataset generation completed!")
    print(f"Total images generated: {sum(len(files) for files in synthetic_files.values())}")

if __name__ == "__main__":
    main() 