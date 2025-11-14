import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import joblib

print("📊 Evaluating saved model...")

# Load the saved model and label encoder
model = tf.keras.models.load_model('best_cattle_disease_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Load your test generator (you'll need to recreate it)
def create_test_generator():
    # You'll need to adapt this part based on your original data
    # This is a simplified version - you might need your original test_df
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # You need to provide the path to your test images
    # Replace this with your actual test data path
    test_generator = test_datagen.flow_from_directory(
        'combined',  # or your test directory
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
        subset='validation'  # or however you split your data
    )
    return test_generator

# If you have the test dataframe saved, use this instead:
def evaluate_with_dataframe():
    print("🔧 Loading test data...")
    
    # Load your test dataframe (if you saved it)
    # test_df = pd.read_csv('test_data.csv')  # if you saved it
    
    # Or recreate the test generator like in your original code
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # You need to modify this to match your test data structure
    test_generator = test_datagen.flow_from_directory(
        'combined',  # Your dataset path
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

try:
    # Try to create test generator
    test_gen = evaluate_with_dataframe()
    
    print("🎯 Evaluating model...")
    
    # Simple evaluation
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    # Detailed predictions
    test_gen.reset()
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes
    
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    
    print(f"✅ Test Precision: {precision:.4f}")
    print(f"✅ Test Recall: {recall:.4f}")
    
    print("\n📈 Classification Report:")
    print(classification_report(y_true, y_pred_classes, 
                              target_names=label_encoder.classes_))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 FINAL RESULTS: {test_accuracy*100:.2f}% Accuracy!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Alternative: Just check the model files exist:")
    import os
    print(f"   Model file exists: {os.path.exists('best_cattle_disease_model.h5')}")
    print(f"   Label encoder exists: {os.path.exists('label_encoder.pkl')}")