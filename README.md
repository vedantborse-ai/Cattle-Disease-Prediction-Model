# 🐄 Cattle Skin Disease Classification using Deep Learning

---

## 📝 Overview

This project is a **Deep Learning–based Cattle Skin Disease Classification System** that identifies multiple cattle skin diseases from images.  
It uses **MobileNetV2**, **ResNet50**, **NASNetMobile**, and a **Custom CNN**, trained on a folder-based image dataset using **TensorFlow/Keras**.

The system supports **automatic preprocessing**, **augmentation**, **class balancing**, and **model evaluation**, and saves all trained artifacts for reuse.

---

## 🚀 Features

✨ **Automatic dataset loading** from folder-structured classes  
🧪 **Train / Validation / Test split** (70% / 15% / 15%)  
🏷️ **Label encoding** for disease classes  
🖼️ **Image preprocessing + augmentation**  
🤖 **Multiple model architectures:**
- MobileNetV2 (default)
- ResNet50
- NASNetMobile
- Custom CNN

⚖️ **Handles class imbalance** using class weights  
🛠️ **Training callbacks included:**
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

📊 **Evaluation includes:**
- Accuracy
- Classification Report
- Confusion Matrix (saved as `.png`)

💾 **Saves the following outputs:**
- Trained model (`cattle_skin_disease_model.h5`)
- Label encoder (`label_encoder.pkl`)
- Confusion matrix image (`confusion_matrix.png`)

---

## 🛠️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| **Language** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **Preprocessing** | OpenCV, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Utilities** | Scikit-learn, Pickle |

---

## 📂 Project Structure

```bash
📁 Cattle-Skin-Disease-Classification/
│
├── combined/                       # Dataset folder (each subfolder = class)
│   ├── Disease1/
│   ├── Disease2/
│   ├── Disease3/
│   └── ...
│
├── train.py                        # Main training script
├── cattle_skin_disease_model.h5    # Saved trained model
├── label_encoder.pkl               # Saved label encoder for predictions
├── confusion_matrix.png            # Saved confusion matrix visualization
└── README.md                       # Documentation
```

---

## 🧩 How It Works

1. 📥 **Dataset Loading**  
   All images are loaded from the `combined/` directory where each subfolder represents a disease.

2. 🖼️ **Preprocessing & Augmentation**  
   Images are resized, normalized, and augmented for robustness.

3. 🤖 **Model Selection & Training**  
   The model is trained using MobileNetV2.  

4. 🧪 **Evaluation**  
   The model outputs accuracy, a classification report, and a confusion matrix.

5. 💾 **Saving Model Artifacts**  
   The trained model, label encoder, and confusion matrix plot are stored for later inference.

---

## 🎯 Future Enhancements

🚀 Add lesion detection using **YOLO/Detectron2**  
📱 Mobile app integration for field cattle diagnosis  
📈 Deploy as a **web app** using Streamlit or FastAPI  
🧬 Add **Vision Transformers (ViT)** for higher accuracy  
🌐 Expand dataset with real-world variations  

---

## ❤️ Acknowledgements

Special thanks to dataset providers supporting livestock disease identification.
```
