import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import os
from tkinter import Tk, filedialog
from tkinter import messagebox

print("🔮 Cattle Skin Disease Prediction System")
print("="*50)

class CattleDiseasePredictor:
    def __init__(self, model_path='best_cattle_disease_model.h5', encoder_path='label_encoder.pkl'):
        print("🔄 Loading model and label encoder...")
        self.model = tf.keras.models.load_model(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.img_height = 224
        self.img_width = 224
        print(f"✅ Model loaded successfully!")
        print(f"🏷️ Available classes: {list(self.label_encoder.classes_)}")
    
    def predict_disease(self, image_path):
        """Predict disease from a single image"""
        try:
            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(self.img_height, self.img_width)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
            img_array /= 255.0  # Normalize
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            disease = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Get all probabilities
            all_probs = {}
            for i, prob in enumerate(predictions[0]):
                disease_name = self.label_encoder.inverse_transform([i])[0]
                all_probs[disease_name] = float(prob)
            
            return disease, confidence, all_probs, img
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None, None, None, None
    
    def display_prediction(self, image_path):
        """Display prediction with visualization"""
        disease, confidence, all_probs, img = self.predict_disease(image_path)
        
        if disease is None:
            return
        
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"🩺 Predicted Disease: {disease}")
        print(f"🎯 Confidence: {confidence:.2%}")
        
        # Display image and prediction
        plt.figure(figsize=(12, 5))
        
        # Image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f'Input Image\nPredicted: {disease}\nConfidence: {confidence:.2%}', 
                 fontsize=12, pad=20)
        plt.axis('off')
        
        # Probabilities bar chart
        plt.subplot(1, 2, 2)
        diseases = list(all_probs.keys())
        probs = [all_probs[d] for d in diseases]
        
        colors = ['red' if d == disease else 'blue' for d in diseases]
        bars = plt.barh(diseases, probs, color=colors)
        plt.xlabel('Probability')
        plt.title('Disease Probabilities')
        plt.xlim(0, 1)
        
        # Add probability values on bars
        for bar, prob in zip(bars, probs):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.2%}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
        
        return disease, confidence

def select_image_file():
    """Open file dialog to select an image"""
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title="Select Cattle Skin Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def main():
    # Initialize predictor
    predictor = CattleDiseasePredictor()
    
    print("\n🔮 Single Image Prediction System")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Select and predict an image")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == '1':
            print("\n📁 Please select an image file...")
            image_path = select_image_file()
            
            if image_path:
                print(f"✅ Selected: {os.path.basename(image_path)}")
                predictor.display_prediction(image_path)
            else:
                print("❌ No file selected!")
                
        elif choice == '2':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice! Please enter 1 or 2.")

if __name__ == "__main__":
    main()