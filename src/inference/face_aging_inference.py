import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import json
from typing import Tuple, List, Optional, Dict, Any
import logging
from pathlib import Path

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.face_aging_model import FaceAgingModel, create_face_aging_model
from data.preprocessing import FacePreprocessor, AgeGroupMapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceAgingInference:
    """Complete inference pipeline for face aging"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 image_size: int = 128):
        self.device = device
        self.image_size = image_size
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize preprocessor
        self.preprocessor = FacePreprocessor(
            target_size=image_size,
            use_face_detection=True,
            use_face_alignment=True
        )
        
        # Initialize age mapper
        self.age_mapper = AgeGroupMapper(num_groups=10)
        
        # Age group descriptions
        self.age_descriptions = [
            "Child (0-10)", "Teen (11-20)", "Young Adult (21-30)", 
            "Adult (31-40)", "Middle Age (41-50)", "Senior (51-60)",
            "Elderly (61-70)", "Senior Citizen (71-80)", 
            "Elder (81-90)", "Centenarian (91-100)"
        ]
    
    def _load_model(self, model_path: str) -> FaceAgingModel:
        """Load the trained face aging model"""
        try:
            # Create model
            model = create_face_aging_model(
                input_size=self.image_size,
                latent_dim=100,
                age_embedding_dim=64,
                num_age_groups=10
            )
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_input(self, image_path: str) -> Optional[torch.Tensor]:
        """Preprocess input image for inference"""
        try:
            processed = self.preprocessor.preprocess_image(image_path, augment=False)
            if processed is not None:
                return processed.unsqueeze(0)  # Add batch dimension
            return None
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def age_face(self, 
                 image_tensor: torch.Tensor, 
                 target_age_group: int) -> torch.Tensor:
        """Age a face to the specified age group"""
        with torch.no_grad():
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Create target age tensor
            target_age = torch.tensor([target_age_group], device=self.device)
            
            # Encode
            z = self.model.encode(image_tensor)
            
            # Decode with target age
            aged_face = self.model.decode(z, target_age)
            
            return aged_face
    
    def generate_age_progression(self, 
                                image_path: str, 
                                target_ages: List[int] = None) -> Dict[str, Any]:
        """Generate age progression for a face"""
        if target_ages is None:
            target_ages = [0, 2, 4, 6, 8]  # Default age groups
        
        try:
            # Preprocess input image
            input_tensor = self.preprocess_input(image_path)
            if input_tensor is None:
                return {"error": "Could not preprocess input image"}
            
            # Generate aged versions
            aged_faces = {}
            original_face = self.preprocessor.tensor_to_image(input_tensor.squeeze(0))
            
            for age_group in target_ages:
                aged_tensor = self.age_face(input_tensor, age_group)
                aged_image = self.preprocessor.tensor_to_image(aged_tensor.squeeze(0))
                
                age_range = self.age_mapper.group_to_age_range(age_group)
                aged_faces[f"age_{age_range[0]}-{age_range[1]}"] = {
                    "image": aged_image,
                    "age_group": age_group,
                    "description": self.age_descriptions[age_group]
                }
            
            return {
                "original": original_face,
                "aged_faces": aged_faces,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating age progression: {str(e)}")
            return {"error": str(e), "success": False}
    
    def save_results(self, 
                     results: Dict[str, Any], 
                     output_dir: str,
                     base_filename: str = "aged_face") -> List[str]:
        """Save age progression results"""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        try:
            # Save original
            original_path = os.path.join(output_dir, f"{base_filename}_original.jpg")
            results["original"].save(original_path)
            saved_paths.append(original_path)
            
            # Save aged faces
            for age_key, age_data in results["aged_faces"].items():
                age_path = os.path.join(output_dir, f"{base_filename}_{age_key}.jpg")
                age_data["image"].save(age_path)
                saved_paths.append(age_path)
            
            # Create comparison image
            comparison_path = self._create_comparison_image(results, output_dir, base_filename)
            if comparison_path:
                saved_paths.append(comparison_path)
            
            logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return saved_paths
    
    def _create_comparison_image(self, 
                                results: Dict[str, Any], 
                                output_dir: str,
                                base_filename: str) -> Optional[str]:
        """Create a side-by-side comparison image"""
        try:
            # Get all images
            images = [results["original"]]
            labels = ["Original"]
            
            for age_key, age_data in results["aged_faces"].items():
                images.append(age_data["image"])
                labels.append(age_data["description"])
            
            # Calculate grid dimensions
            n_images = len(images)
            cols = min(5, n_images)
            rows = (n_images + cols - 1) // cols
            
            # Create comparison image
            img_width, img_height = images[0].size
            comparison_width = cols * img_width
            comparison_height = rows * img_height + 30 * rows  # Extra space for labels
            
            comparison_img = Image.new('RGB', (comparison_width, comparison_height), 'white')
            
            # Paste images and add labels
            for i, (img, label) in enumerate(zip(images, labels)):
                row = i // cols
                col = i % cols
                
                x = col * img_width
                y = row * (img_height + 30)
                
                comparison_img.paste(img, (x, y))
                
                # Add label
                draw = ImageDraw.Draw(comparison_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                # Calculate text position
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = x + (img_width - text_width) // 2
                text_y = y + img_height + 5
                
                draw.text((text_x, text_y), label, fill='black', font=font)
            
            # Save comparison image
            comparison_path = os.path.join(output_dir, f"{base_filename}_comparison.jpg")
            comparison_img.save(comparison_path)
            
            return comparison_path
            
        except Exception as e:
            logger.error(f"Error creating comparison image: {str(e)}")
            return None
    
    def batch_process(self, 
                      image_paths: List[str], 
                      output_dir: str,
                      target_ages: List[int] = None) -> Dict[str, Any]:
        """Process multiple images in batch"""
        results = {
            "processed": [],
            "failed": [],
            "total": len(image_paths)
        }
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Generate age progression
                age_results = self.generate_age_progression(image_path, target_ages)
                
                if age_results.get("success", False):
                    # Save results
                    base_filename = Path(image_path).stem
                    saved_paths = self.save_results(age_results, output_dir, base_filename)
                    
                    results["processed"].append({
                        "input_path": image_path,
                        "output_paths": saved_paths,
                        "age_results": age_results
                    })
                else:
                    results["failed"].append({
                        "input_path": image_path,
                        "error": age_results.get("error", "Unknown error")
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                results["failed"].append({
                    "input_path": image_path,
                    "error": str(e)
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": "Face Aging CAAE",
            "input_size": self.image_size,
            "latent_dim": self.model.latent_dim,
            "age_embedding_dim": self.model.age_embedding_dim,
            "num_age_groups": self.model.num_age_groups,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": self.device,
            "age_groups": self.age_descriptions
        }


class FaceAgingApp:
    """Simple application interface for face aging"""
    
    def __init__(self, model_path: str):
        self.inference = FaceAgingInference(model_path)
    
    def run_interactive(self):
        """Run interactive face aging application"""
        print("=== Face Aging Model ===")
        print("Upload an image to see how you might look at different ages!")
        
        while True:
            print("\nOptions:")
            print("1. Process single image")
            print("2. Batch process images")
            print("3. Show model info")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                self._process_single_image()
            elif choice == "2":
                self._batch_process_images()
            elif choice == "3":
                self._show_model_info()
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _process_single_image(self):
        """Process a single image interactively"""
        image_path = input("Enter the path to your image: ").strip()
        
        if not os.path.exists(image_path):
            print("Image not found!")
            return
        
        print("Processing image...")
        
        # Generate age progression
        results = self.inference.generate_age_progression(image_path)
        
        if results.get("success", False):
            # Save results
            output_dir = "output/aged_faces"
            base_filename = Path(image_path).stem
            saved_paths = self.inference.save_results(results, output_dir, base_filename)
            
            print(f"Results saved to: {output_dir}")
            print("Generated ages:")
            for age_key, age_data in results["aged_faces"].items():
                print(f"  - {age_data['description']}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
    
    def _batch_process_images(self):
        """Batch process images interactively"""
        input_dir = input("Enter the directory containing images: ").strip()
        
        if not os.path.exists(input_dir):
            print("Directory not found!")
            return
        
        # Get image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_paths:
            print("No images found in directory!")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Process images
        output_dir = "output/batch_aged_faces"
        results = self.inference.batch_process(image_paths, output_dir)
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {len(results['processed'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Results saved to: {output_dir}")
    
    def _show_model_info(self):
        """Show model information"""
        info = self.inference.get_model_info()
        
        print("\n=== Model Information ===")
        for key, value in info.items():
            print(f"{key}: {value}")


def create_inference_pipeline(model_path: str,
                             device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> FaceAgingInference:
    """Factory function to create inference pipeline"""
    return FaceAgingInference(model_path, device)


if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/face_aging_model_epoch_100.pth"
    
    if os.path.exists(model_path):
        # Create inference pipeline
        inference = create_inference_pipeline(model_path)
        
        # Create app
        app = FaceAgingApp(model_path)
        app.run_interactive()
    else:
        print(f"Model not found at {model_path}")
        print("Please train the model first or provide the correct model path.") 