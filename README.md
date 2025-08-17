# Maize Leaf Disease Classification

A modern web app for classifying maize leaf diseases using Vision Transformers (ViT). Built with Streamlit and HuggingFace Transformers.

## 🚀 Live Demo

- **App:** [maize-disease-classification.streamlit.app](https://maize-disease-classification.streamlit.app)
- **Model Training Notebook:** [vision_transformers_maize_leaf_disease_detection_s.ipynb](https://github.com/ogbuaguwizard/maize-disease-classification/blob/main/model_training/vision_transformers_maize_leaf_disease_detection_s.ipynb)

---

## 🛠️ How to Clone and Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ogbuaguwizard/maize-disease-classification.git
   cd maize-disease-classification
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the model files:**
   - Place your local model files in a folder named `maize_vit_model` in the project root (if running offline).
   - Or, the app will automatically download from HuggingFace if not found locally.

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

6. **Open in your browser:**
   - Go to [http://localhost:8501](http://localhost:8501)

---

## 📝 About

- **Frontend:** Streamlit (with custom CSS for a modern UI)
- **Model:** Vision Transformer (ViT) fine-tuned for maize leaf disease detection
- **Model Training:** See the [training notebook](https://github.com/ogbuaguwizard/maize-disease-classification/blob/main/model_training/vision_transformers_maize_leaf_disease_detection_s.ipynb)

---

## 📄 License

This project is open-source and free to use for research and educational purposes.
