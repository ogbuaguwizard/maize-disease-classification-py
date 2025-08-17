import streamlit as st
from PIL import Image
import torch
from transformers import ViTForImageClassification, AutoImageProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model & processor
@st.cache_resource
def load_model():
    try:
        model_path = "francis-ogbuagu/maize_vit_model"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ViTForImageClassification.from_pretrained(model_path, use_safetensors=True)
        processor = AutoImageProcessor.from_pretrained(model_path)

        model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully on {device}")
        return model, processor, device
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None, None, None

model, processor, device = load_model()

# Class labels
labels_map = {
    0: "Common Rust",
    1: "Gray Leaf Spot",
    2: "Blight",
    3: "Healthy"
}

# CSS for styling - responsive design
st.markdown("""
<style>
/* Responsive header */
.app-header {
  background: #1e293b;
  color: #fff;
  padding: 1.5rem 1rem;
  text-align: center;
  border-bottom-left-radius: 1.5rem;
  border-bottom-right-radius: 1.5rem;
  margin-bottom: 1.5rem;
}
.app-header h1 {
  font-size: 1.8rem;
  margin-bottom: 0.5rem;
}
.app-header p {
  font-size: 1rem;
  opacity: 0.9;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .app-header h1 {
    font-size: 1.5rem;
  }
  .app-header {
    padding: 1rem;
  }
}

/* Prediction cards */
.prediction-card {
  background: #1e293b;
  color: #fff;
  border-radius: 0.5rem;
  padding: 0.75rem 1rem;
  margin: 0.5rem 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.progress-container {
  flex: 2;
  background: #334155;
  border-radius: 4px;
  height: 12px;
  margin: 0 10px;
}
.progress-bar {
  background: #facc15;
  height: 100%;
  border-radius: 4px;
}

/* Image styling */
.stImage>img {
  border-radius: 0.5rem;
  margin-bottom: 1rem;
  max-width: 100%;
}

/* Button styling */
.stButton>button {
  width: 200px;
  max-width: 100%;
  background-color: #4caf50;
  color: #fff;
  border: none;
  border-radius: 0.5rem;
  padding: 0.75rem;
  font-size: 1.1rem;
  font-weight: 700;
  cursor: pointer;
  transition: background-color 0.2s;
  display: block;
  margin: 0 auto;
}
.stButton>button:hover {
  background-color: #45a049;
}
.stButton>button:active {
  background-color: #388e3c;
}

/* File uploader styling */
.upload-section {
  text-align: center;
  margin-bottom: 1.5rem;
}
.upload-section .stFileUploader > label {
  display: none !important;
}
.upload-title {
  font-size: 1.4rem;
  margin-bottom: 0.5rem;
  color: #1e293b;
  font-weight: 600;
}
.upload-subtitle {
  color: #64748b;
  margin-bottom: 1rem;
}

/* Responsive columns */
@media (max-width: 768px) {
  .column-container {
    flex-direction: column;
  }
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="app-header">'
    '<h1>MAIZE LEAF DISEASE DETECTION</h1>'
    '<p>Powered by Vision Transformers</p>'
    '</div>', 
    unsafe_allow_html=True
)

if model is None:
    st.stop()

# Upload section - improved layout
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<div class="upload-title">Upload Leaf Image</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-subtitle">For disease analysis and diagnosis</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    " ",  # Empty label since we have our own title
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        if image.size[0] < 50 or image.size[1] < 50:
            st.error("Image too small. Please upload a larger one.")
            st.stop()

        # Responsive column layout
        st.markdown('<div class="column-container" style="display: flex; flex-wrap: wrap; gap: 2rem;">', unsafe_allow_html=True)
        
        # Image column
        st.markdown('<div style="flex: 1; min-width: 300px;">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Centered button
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            analyze = st.button("Analyze Image", key="analyze", help="Click to analyze the leaf")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close image column
        
        # Results column
        st.markdown('<div style="flex: 1; min-width: 300px;">', unsafe_allow_html=True)
        
        if analyze:
            with st.spinner("Analyzing image..."):
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    pred_class = logits.argmax(-1).item()
                    confidence = torch.softmax(logits, dim=1)[0][pred_class].item()

            st.success("✅ Analysis complete!")
            st.markdown("### Prediction & Confidence")
            st.markdown(f"""
                <div class="prediction-card">
                    <span><b>{labels_map[pred_class]}</b></span>
                    <span><b>{confidence:.2%}</b></span>
                </div>
            """, unsafe_allow_html=True)

            st.subheader("Confidence Breakdown")
            probabilities = torch.softmax(logits, dim=1)[0]
            for i, prob in enumerate(probabilities):
                st.markdown(f"""
                <div class="prediction-card">
                    <span>{labels_map[i]}</span>
                    <div class="progress-container">
                        <div class="progress-bar" style="width:{prob*100:.1f}%"></div>
                    </div>
                    <span>{prob:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close results column
        st.markdown('</div>', unsafe_allow_html=True)  # Close column container

    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")