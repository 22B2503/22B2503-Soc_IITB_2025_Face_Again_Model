"""
Data Preprocessing Utilities for Face Aging Model
Handles face detection, alignment, and augmentation
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import mediapipe as mp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from typing import Tuple, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection and alignment using MediaPipe"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        # Initialize face mesh for landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """Detect faces in the image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': detection.score[0],
                    'keypoints': detection.location_data.relative_keypoints
                })
        
        return faces
    
    def get_face_landmarks(self, image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Get facial landmarks for alignment"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w, _ = image.shape
            
            # Extract key landmarks for alignment
            key_landmarks = []
            landmark_indices = [33, 133, 362, 263, 61, 291]  # Eyes, nose, mouth corners
            
            for idx in landmark_indices:
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    key_landmarks.append((x, y))
            
            return key_landmarks
        
        return None
    
    def align_face(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Align face using landmarks"""
        if len(landmarks) < 4:
            return image
        
        # Get eye landmarks (assuming first two landmarks are eyes)
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Calculate angle for alignment
        eye_angle = np.degrees(np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        ))
        
        # Get center point between eyes
        eye_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2
        )
        
        # Rotate image
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, eye_angle)
        aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        
        return aligned_image
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                  margin: float = 0.2) -> np.ndarray:
        """Crop face with margin"""
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]
        
        # Add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # Calculate crop coordinates
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)
        
        return image[y1:y2, x1:x2]


class FacePreprocessor:
    """Complete face preprocessing pipeline"""
    
    def __init__(self, 
                 target_size: int = 128,
                 use_face_detection: bool = True,
                 use_face_alignment: bool = True):
        self.target_size = target_size
        self.use_face_detection = use_face_detection
        self.use_face_alignment = use_face_alignment
        
        if use_face_detection:
            self.face_detector = FaceDetector()
        
        # Define augmentation pipeline
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.Resize(target_size, target_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
        
        # Define validation pipeline (no augmentation)
        self.validation_transform = A.Compose([
            A.Resize(target_size, target_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    
    def preprocess_image(self, 
                        image_path: str, 
                        augment: bool = False) -> Optional[torch.Tensor]:
        """Preprocess a single image"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Could not load image: {image_path}")
                    return None
            else:
                image = image_path
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Face detection and alignment
            if self.use_face_detection:
                faces = self.face_detector.detect_faces(image)
                
                if not faces:
                    logger.warning(f"No faces detected in image: {image_path}")
                    return None
                
                # Use the first detected face
                face = faces[0]
                image = self.face_detector.crop_face(image, face['bbox'])
                
                # Face alignment
                if self.use_face_alignment:
                    landmarks = self.face_detector.get_face_landmarks(image)
                    if landmarks:
                        image = self.face_detector.align_face(image, landmarks)
            
            # Apply transformations
            if augment:
                transformed = self.augmentation(image=image)
            else:
                transformed = self.validation_transform(image=image)
            
            return transformed['image']
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def preprocess_batch(self, 
                        image_paths: List[str], 
                        augment: bool = False) -> List[torch.Tensor]:
        """Preprocess a batch of images"""
        processed_images = []
        
        for image_path in image_paths:
            processed = self.preprocess_image(image_path, augment)
            if processed is not None:
                processed_images.append(processed)
        
        return processed_images
    
    def create_face_dataset(self, 
                           data_dir: str, 
                           output_dir: str,
                           augment: bool = False) -> List[str]:
        """Create a processed face dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        processed_paths = []
        
        # Walk through data directory
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    input_path = os.path.join(root, file)
                    
                    # Preprocess image
                    processed = self.preprocess_image(input_path, augment)
                    
                    if processed is not None:
                        # Save processed image
                        output_path = os.path.join(
                            output_dir, 
                            f"processed_{os.path.basename(input_path)}"
                        )
                        
                        # Convert tensor back to image and save
                        processed_img = self.tensor_to_image(processed)
                        processed_img.save(output_path)
                        
                        processed_paths.append(output_path)
        
        logger.info(f"Processed {len(processed_paths)} images")
        return processed_paths
    
    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL Image"""
        # Denormalize
        tensor = tensor * 0.5 + 0.5
        
        # Convert to numpy
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        image_array = tensor.permute(1, 2, 0).numpy()
        image_array = (image_array * 255).astype(np.uint8)
        
        return Image.fromarray(image_array)
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancement techniques"""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Enhance color
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        return image


class AgeGroupMapper:
    """Map ages to age groups"""
    
    def __init__(self, num_groups: int = 10):
        self.num_groups = num_groups
        self.age_ranges = self._create_age_ranges()
    
    def _create_age_ranges(self) -> List[Tuple[int, int]]:
        """Create age ranges for groups"""
        if self.num_groups == 10:
            return [
                (0, 10), (11, 20), (21, 30), (31, 40), (41, 50),
                (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)
            ]
        elif self.num_groups == 5:
            return [
                (0, 20), (21, 40), (41, 60), (61, 80), (81, 100)
            ]
        else:
            # Create custom ranges
            step = 100 // self.num_groups
            ranges = []
            for i in range(self.num_groups):
                start = i * step
                end = (i + 1) * step if i < self.num_groups - 1 else 100
                ranges.append((start, end))
            return ranges
    
    def age_to_group(self, age: int) -> int:
        """Convert age to group index"""
        for i, (min_age, max_age) in enumerate(self.age_ranges):
            if min_age <= age <= max_age:
                return i
        return len(self.age_ranges) - 1  # Default to last group
    
    def group_to_age_range(self, group_idx: int) -> Tuple[int, int]:
        """Convert group index to age range"""
        if 0 <= group_idx < len(self.age_ranges):
            return self.age_ranges[group_idx]
        return (0, 100)  # Default range
    
    def get_group_center_age(self, group_idx: int) -> int:
        """Get the center age for a group"""
        min_age, max_age = self.group_to_age_range(group_idx)
        return (min_age + max_age) // 2


def create_preprocessing_pipeline(target_size: int = 128,
                                 use_face_detection: bool = True,
                                 use_face_alignment: bool = True) -> FacePreprocessor:
    """Factory function to create preprocessing pipeline"""
    return FacePreprocessor(
        target_size=target_size,
        use_face_detection=use_face_detection,
        use_face_alignment=use_face_alignment
    )


if __name__ == "__main__":
    # Example usage
    preprocessor = create_preprocessing_pipeline()
    
    # Test with a sample image
    # processed = preprocessor.preprocess_image("path/to/image.jpg")
    # if processed is not None:
    #     print(f"Processed image shape: {processed.shape}")
    
    # Test age mapping
    age_mapper = AgeGroupMapper()
    print(f"Age 25 -> Group {age_mapper.age_to_group(25)}")
    print(f"Age 65 -> Group {age_mapper.age_to_group(65)}")
    print(f"Group 3 -> Age range {age_mapper.group_to_age_range(3)}") 