# CIFAR-10 Image Classifier

A CNN-based image classifier trained on the CIFAR-10 dataset with a web interface for real-time predictions.

## 🌐 Live Demo

**Try it online:** [CIFAR-10 Classifier on Streamlit](link-after-deployment)

Upload any image and get instant predictions across 10 object categories!

## Problem Statement

Classify 32x32 pixel images into 10 categories: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Dataset

- **CIFAR-10** from TensorFlow/Keras
- 60,000 training images (50,000) + test images (10,000)
- 10 object categories
- 32x32 RGB pixel images

## Model Architecture
Conv2D(32) → BatchNorm → MaxPool
Conv2D(64) → BatchNorm → MaxPool
Conv2D(128) → BatchNorm → MaxPool
Dense(128, relu) → Dropout(0.5)
Dense(10, softmax)

## Results

- **Test Accuracy:** 75%+ (depends on training)
- **Data Augmentation:** Rotation, shift, flip, zoom
- **Training Time:** ~45-60 minutes on CPU
- **Parameters:** ~400K

## Tech Stack

- TensorFlow / Keras
- NumPy, Matplotlib
- Streamlit (web interface)
- Python 3.8+

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
python train.py
```

This will:
- Download CIFAR-10 dataset
- Train CNN for 30 epochs
- Save `cifar10_model.h5`
- Generate visualizations

### 3. Run Web Interface
```bash
streamlit run app.py
```

Open http://localhost:8501 and upload images!

## Project Structure
```text
├── train.py                # Training script
├── app.py                  # Streamlit web app
├── requirements.txt        # Dependencies
├── cifar10_model.h5       # Trained model
├── training_history.png   # Accuracy/loss plots
├── sample_predictions.png # Example predictions
└── README.md              # This file
```
## Features

✅ **Real-time Predictions** — Upload images and get instant classifications  
✅ **Confidence Scores** — See probability distribution  
✅ **Training Visualizations** — Accuracy and loss curves  
✅ **Data Augmentation** — Improves model generalization  
✅ **Batch Normalization** — Faster convergence  
✅ **Interactive Web UI** — No coding required  

## Key Learnings

- CNNs excel at image classification
- Batch normalization stabilizes training
- Data augmentation prevents overfitting
- Dropout reduces overfitting with larger models
- Streamlit enables rapid deployment

## Future Improvements

- Implement ResNet or EfficientNet for higher accuracy
- Add model pruning for mobile deployment
- Implement class activation maps (CAM) for interpretability
- Train on ImageNet for transfer learning
- Add webcam support for live classification

## About

A computer vision project demonstrating CNN architecture, training best practices, and production deployment.
