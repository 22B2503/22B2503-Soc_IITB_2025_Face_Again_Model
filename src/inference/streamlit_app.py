import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
import io
from pathlib import Path
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.face_aging_model import create_face_aging_model
from data.preprocessing import FacePreprocessor, AgeGroupMapper
from inference.face_aging_inference import FaceAgingInference

# Page configuration
st.set_page_config(
    page_title="Face Aging AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
    }
    .age-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 5px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'inference' not in st.session_state:
    st.session_state.inference = None

def load_model():
    """Load the face aging model"""
    try:
        model_path = "checkpoints/face_aging_model_best.pth"
        
        if not os.path.exists(model_path):
            # Try alternative paths
            checkpoint_files = [f for f in os.listdir("checkpoints") if f.endswith('.pth')]
            if checkpoint_files:
                model_path = os.path.join("checkpoints", checkpoint_files[-1])  # Use latest checkpoint
            else:
                st.error("No model checkpoint found! Please train the model first.")
                return False
        
        # Create inference pipeline
        inference = FaceAgingInference(model_path)
        st.session_state.inference = inference
        st.session_state.model_loaded = True
        
        return True
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False

def process_image(image, target_ages):
    """Process image and generate aged versions"""
    try:
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        # Generate age progression
        results = st.session_state.inference.generate_age_progression(tmp_path, target_ages)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return results
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def create_comparison_image(results):
    """Create a comparison image with all age versions"""
    if not results or not results.get("success"):
        return None
    
    # Get images
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
    comparison_height = rows * img_height + 40 * rows  # Extra space for labels
    
    comparison_img = Image.new('RGB', (comparison_width, comparison_height), 'white')
    
    # Paste images and add labels
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols
        
        x = col * img_width
        y = row * (img_height + 40)
        
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
        text_y = y + img_height + 10
        
        # Draw text with background
        draw.rectangle([text_x-5, text_y-5, text_x+text_width+5, text_y+20], fill='black')
        draw.text((text_x, text_y), label, fill='white', font=font)
    
    return comparison_img

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 Face Aging AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">See how you might look at different ages with our advanced AI model</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading
        if not st.session_state.model_loaded:
            if st.button("Load Model"):
                with st.spinner("Loading model..."):
                    if load_model():
                        st.success("Model loaded successfully!")
                    else:
                        st.error("Failed to load model")
        else:
            st.success("✅ Model loaded")
            
            # Model info
            if st.session_state.inference:
                info = st.session_state.inference.get_model_info()
                st.subheader("Model Information")
                st.write(f"**Type:** {info['model_type']}")
                st.write(f"**Parameters:** {info['total_parameters']:,}")
                st.write(f"**Device:** {info['device']}")
        
        # Age selection
        st.subheader("🎂 Age Groups")
        age_descriptions = [
            "Child (0-10)", "Teen (11-20)", "Young Adult (21-30)", 
            "Adult (31-40)", "Middle Age (41-50)", "Senior (51-60)",
            "Elderly (61-70)", "Senior Citizen (71-80)", 
            "Elder (81-90)", "Centenarian (91-100)"
        ]
        
        selected_ages = []
        for i, desc in enumerate(age_descriptions):
            if st.checkbox(desc, value=i in [0, 2, 4, 6, 8]):  # Default selection
                selected_ages.append(i)
        
        if not selected_ages:
            st.warning("Please select at least one age group")
        
        # Processing options
        st.subheader("🔧 Options")
        show_comparison = st.checkbox("Show comparison grid", value=True)
        save_results = st.checkbox("Save results", value=False)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📸 Upload Your Photo")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear photo of a face. The model works best with front-facing portraits."
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🚀 Generate Age Progression", disabled=not selected_ages):
                if not st.session_state.model_loaded:
                    st.error("Please load the model first!")
                else:
                    with st.spinner("Processing image..."):
                        results = process_image(image, selected_ages)
                        
                        if results and results.get("success"):
                            st.session_state.results = results
                            st.success("Age progression generated successfully!")
                        else:
                            st.error("Failed to generate age progression")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.subheader("🎯 Results")
        
        if 'results' in st.session_state and st.session_state.results:
            results = st.session_state.results
            
            # Show individual age versions
            if not show_comparison:
                for age_key, age_data in results["aged_faces"].items():
                    st.write(f"**{age_data['description']}**")
                    st.image(age_data["image"], use_column_width=True)
            
            # Show comparison grid
            if show_comparison:
                comparison_img = create_comparison_image(results)
                if comparison_img:
                    st.image(comparison_img, caption="Age Progression Comparison", use_column_width=True)
                    
                    # Download button
                    if save_results:
                        buf = io.BytesIO()
                        comparison_img.save(buf, format='PNG')
                        buf.seek(0)
                        st.download_button(
                            label="📥 Download Comparison",
                            data=buf.getvalue(),
                            file_name="age_progression_comparison.png",
                            mime="image/png"
                        )
            
            # Individual download options
            if save_results:
                st.subheader("📥 Download Individual Images")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # Original
                    buf = io.BytesIO()
                    results["original"].save(buf, format='PNG')
                    buf.seek(0)
                    st.download_button(
                        label="Original",
                        data=buf.getvalue(),
                        file_name="original.png",
                        mime="image/png"
                    )
                
                with col_b:
                    # Aged versions
                    for age_key, age_data in results["aged_faces"].items():
                        buf = io.BytesIO()
                        age_data["image"].save(buf, format='PNG')
                        buf.seek(0)
                        st.download_button(
                            label=age_data["description"],
                            data=buf.getvalue(),
                            file_name=f"aged_{age_key}.png",
                            mime="image/png"
                        )
        else:
            st.info("Upload an image and click 'Generate Age Progression' to see results")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🧬 Face Aging AI - Powered by Deep Learning</p>
        <p>This model uses Conditional Adversarial Autoencoders (CAAE) to generate realistic age progressions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 