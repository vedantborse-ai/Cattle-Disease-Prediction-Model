# 🐄 Cattle Skin Disease Classification using Deep Learning

This project builds a deep-learning model to classify multiple **cattle skin diseases** from images using MobileNetV2, ResNet50, NASNetMobile, and a custom CNN.  
It uses a folder-based dataset and trains a multi-class classifier with TensorFlow/Keras.

---

## 🚀 Features

- Automatic dataset loading from folders  
- Train/Val/Test split (70/15/15)  
- Label encoding for disease classes  
- Image preprocessing + augmentation  
- Multiple model architectures:
  - MobileNetV2 (default)
  - ResNet50
  - NASNetMobile
  - Custom CNN
- Handles class imbalance with class weights  
- Training callbacks:
  - EarlyStopping  
  - ReduceLROnPlateau  
  - ModelCheckpoint  
- Evaluation:
  - Accuracy  
  - Classification report  
  - Confusion matrix  
- Saves:
  - Trained model (`.h5`)
  - Label encoder (`.pkl`)
  - Confusion matrix image (`.png`)

---

## 📂 Project Structure

project/
│
├── combined/                     # Dataset folder
│   ├── Disease1/
│   ├── Disease2/
│   ├── Disease3/
│   └── ...
│
├── cattle_skin_disease_model.h5  # Saved model
├── label_encoder.pkl             # Saved label encoder
├── confusion_matrix.png          # Saved confusion matrix
│
├── train.py                      # Main training script
└── README.md                     # Documentation
