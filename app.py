import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="CIFAR-10 Classifier", layout="wide")

st.title("🎨 CIFAR-10 Image Classifier")
st.markdown("CNN trained on 60,000 images across 10 object categories")

# Class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load model
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model('cifar10_model.h5')
    except:
        st.error("Model not found. Please train the model first using train.py")
        return None

model = load_model()

if model is not None:
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image Upload", "📊 Model Info", "📈 Training History"])
    
    with tab1:
        st.header("Upload an Image")
        
        uploaded_file = st.file_uploader("Choose an image (32x32 for best results)", 
                                        type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess
            img_resized = image.resize((32, 32))
            img_array = np.array(img_resized).astype('float32') / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_batch, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            with col2:
                st.metric("Predicted Class", class_names[predicted_class])
                st.metric("Confidence", f"{confidence*100:.2f}%")
                
                # Show all probabilities
                st.subheader("All Predictions")
                pred_df = {
                    'Class': class_names,
                    'Confidence': prediction[0]
                }
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(pred_df['Class'], pred_df['Confidence'])
                
                # Color the predicted class
                bars[predicted_class].set_color('green')
                for i, bar in enumerate(bars):
                    if i != predicted_class:
                        bar.set_color('lightblue')
                
                ax.set_xlabel('Confidence')
                ax.set_title('Prediction Probabilities')
                st.pyplot(fig)
    
    with tab2:
        st.header("Model Architecture")
        st.write("""
        **Convolutional Neural Network (CNN) with:**
        - 3 Conv2D layers (32, 64, 128 filters)
        - Batch Normalization after each conv layer
        - MaxPooling for dimensionality reduction
        - Dropout (0.5) for regularization
        - Dense layers for classification
        
        **Training Details:**
        - Dataset: CIFAR-10 (60,000 images)
        - Classes: 10 object categories
        - Data Augmentation: Rotation, shift, flip, zoom
        - Optimizer: Adam
        - Loss: Categorical Cross-Entropy
        - Epochs: 30
        """)
        
        st.subheader("Classes")
        cols = st.columns(2)
        for i, class_name in enumerate(class_names):
            with cols[i % 2]:
                st.write(f"✓ {class_name}")
    
    with tab3:
        st.header("Training History")
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                img1 = Image.open('training_history.png')
                st.image(img1, caption="Accuracy & Loss over Epochs", use_column_width=True)
            
            with col2:
                img2 = Image.open('sample_predictions.png')
                st.image(img2, caption="Sample Predictions", use_column_width=True)
        except:
            st.info("Training visualizations not found. Run train.py first.")