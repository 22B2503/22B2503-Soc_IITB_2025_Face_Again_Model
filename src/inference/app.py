"""
Face Aging Model Web Application
Streamlit app for face aging inference
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import os
import sys
from typing import Optional, Tuple
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.caae import CAAE
from src.data.preprocessing import FacePreprocessor
from utils.visualization import visualize_aging_results

class FaceAgingApp:
    """Streamlit application for face aging inference"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = None
        self.age_options = {
            'Young (20-30)': 0,
            'Middle-aged (40-50)': 1,
            'Elderly (60-70)': 2
        }
        
        # Initialize components
        self._load_model()
        self._setup_preprocessor()
    
    def _load_model(self):
        """Load the trained face aging model"""
        try:
            # Load CAAE model
            model_config = {
                'input_channels': 3,
                'latent_dim': 100,
                'num_age_classes': 3,
                'image_size': 256
            }
            
            self.model = CAAE(**model_config)
            
            # Try to load pre-trained weights
            checkpoint_path = 'checkpoints/best_checkpoint.pth'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                st.success("✅ Pre-trained model loaded successfully!")
            else:
                st.warning("⚠️ No pre-trained model found. Using untrained model for demonstration.")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            # Create a dummy model for demonstration
            self.model = self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration purposes"""
        class DummyModel:
            def __init__(self):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            def to(self, device):
                self.device = device
                return self
            
            def eval(self):
                pass
            
            def __call__(self, x, age_condition):
                # Return a slightly modified version of the input
                return x + torch.randn_like(x) * 0.1
        
        return DummyModel()
    
    def _setup_preprocessor(self):
        """Setup face preprocessing pipeline"""
        self.preprocessor = FacePreprocessor(
            target_size=(256, 256),
            face_detection_method='face_recognition'
        )
    
    def preprocess_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """Preprocess uploaded image"""
        try:
            # Convert PIL to numpy
            image_np = np.array(image)
            
            # Detect and align face
            processed_face = self.preprocessor.preprocess_face(image_np)
            
            if processed_face is None:
                st.error("❌ No face detected in the image. Please upload an image with a clear face.")
                return None
            
            return processed_face
            
        except Exception as e:
            st.error(f"❌ Error preprocessing image: {e}")
            return None
    
    def age_face(self, face_image: np.ndarray, target_age: int) -> np.ndarray:
        """Apply age progression to face image"""
        try:
            # Convert to tensor
            face_tensor = torch.from_numpy(face_image).float()
            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
            face_tensor = face_tensor.to(self.device)
            
            # Normalize to [-1, 1]
            face_tensor = (face_tensor / 255.0) * 2 - 1
            
            # Create age condition
            age_condition = torch.tensor([target_age], device=self.device)
            
            # Generate aged face
            with torch.no_grad():
                aged_face = self.model(face_tensor, age_condition)
            
            # Convert back to numpy
            aged_face = aged_face.squeeze(0).permute(1, 2, 0)
            aged_face = ((aged_face + 1) / 2 * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
            
            return aged_face
            
        except Exception as e:
            st.error(f"❌ Error aging face: {e}")
            return face_image
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Face Aging Model",
            page_icon="🧬",
            layout="wide"
        )
        
        # Header
        st.title("🧬 Face Aging Model")
        st.markdown("Upload a selfie and see how you might look at different ages!")
        
        # Sidebar
        st.sidebar.header("Settings")
        
        # Age selection
        selected_age = st.sidebar.selectbox(
            "Choose target age:",
            list(self.age_options.keys()),
            index=1
        )
        target_age = self.age_options[selected_age]
        
        # Model info
        st.sidebar.header("Model Information")
        st.sidebar.info(f"Device: {self.device}")
        st.sidebar.info(f"Target Age: {selected_age}")
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📸 Upload Your Photo")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear photo of your face"
            )
            
            if uploaded_file is not None:
                # Display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Preprocess image
                processed_face = self.preprocess_image(image)
                
                if processed_face is not None:
                    # Display processed face
                    st.image(processed_face, caption="Processed Face", use_column_width=True)
                    
                    # Age progression button
                    if st.button("🚀 Generate Aged Face", type="primary"):
                        with st.spinner("Generating aged face..."):
                            # Apply age progression
                            aged_face = self.age_face(processed_face, target_age)
                            
                            # Display result
                            with col2:
                                st.header("👴 Aged Face")
                                st.image(aged_face, caption=f"Aged to {selected_age}", use_column_width=True)
                                
                                # Download button
                                aged_image = Image.fromarray(aged_face)
                                buf = io.BytesIO()
                                aged_image.save(buf, format='PNG')
                                st.download_button(
                                    label="📥 Download Aged Face",
                                    data=buf.getvalue(),
                                    file_name=f"aged_face_{target_age}.png",
                                    mime="image/png"
                                )
        
        # Information section
        st.markdown("---")
        st.header("ℹ️ About This Model")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.subheader("🔬 Technology")
            st.markdown("""
            - **CAAE**: Conditional Adversarial Autoencoder
            - **Deep Learning**: PyTorch-based neural networks
            - **Computer Vision**: Advanced face processing
            """)
        
        with col4:
            st.subheader("🎯 Features")
            st.markdown("""
            - **Age Progression**: Realistic aging simulation
            - **Face Detection**: Automatic face alignment
            - **Multiple Ages**: Young, middle-aged, elderly
            - **High Quality**: 256x256 resolution output
            """)
        
        with col5:
            st.subheader("⚠️ Disclaimer")
            st.markdown("""
            This is a demonstration model for educational purposes. 
            Results may vary and should not be considered as accurate predictions.
            """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "Built with ❤️ using PyTorch, Streamlit, and Computer Vision | "
            "[GitHub](https://github.com/your-repo/face-aging-model)"
        )

def main():
    """Main function to run the app"""
    app = FaceAgingApp()
    app.run()

if __name__ == "__main__":
    main() 