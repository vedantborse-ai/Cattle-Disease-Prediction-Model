import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

print("🚀 Starting Cattle Skin Disease Classification Model Training...")

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = "combined"

# 1. Explore Dataset Structure
def explore_dataset_structure(base_path):
    print("📁 Exploring dataset structure...")
    disease_folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            image_files = [f for f in os.listdir(item_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            disease_folders.append({
                'disease': item,
                'path': item_path,
                'image_count': len(image_files)
            })
            print(f"   📂 {item}: {len(image_files)} images")
    
    return disease_folders

disease_info = explore_dataset_structure(DATASET_PATH)
print(f"🎯 Total disease categories: {len(disease_info)}")

# 2. Create DataFrame from Folder Structure
def create_dataframe_from_folders(base_path):
    print("📊 Creating dataset dataframe...")
    data = []
    for disease_folder in os.listdir(base_path):
        disease_path = os.path.join(base_path, disease_folder)
        if os.path.isdir(disease_path):
            for image_file in os.listdir(disease_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        'filename': os.path.join(disease_folder, image_file),
                        'file_path': os.path.join(disease_path, image_file),
                        'disease': disease_folder
                    })
    return pd.DataFrame(data)

# Create the dataframe
df = create_dataframe_from_folders(DATASET_PATH)
print(f"📸 Total images in dataset: {len(df)}")
print("\n📈 Class distribution:")
print(df['disease'].value_counts())

# 3. Data Preprocessing
print("🔧 Preprocessing data...")

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['disease'])

print(f"🎯 Number of classes: {len(label_encoder.classes_)}")
print("🏷️ Classes:", label_encoder.classes_)

# Split the data
train_df, temp_df = train_test_split(df, test_size=0.3, 
                                    stratify=df['label'], 
                                    random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, 
                                  stratify=temp_df['label'], 
                                  random_state=42)

print(f"\n📊 Data split:")
print(f"   Training samples: {len(train_df)}")
print(f"   Validation samples: {len(val_df)}")
print(f"   Test samples: {len(test_df)}")

# 4. Create Data Generators
def create_data_generators(train_df, val_df, test_df, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, batch_size=BATCH_SIZE):
    print("🔄 Creating data generators...")
    
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation and test data (only rescaling)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='file_path',
        y_col='disease',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = test_datagen.flow_from_dataframe(
        val_df,
        x_col='file_path',
        y_col='disease',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col='file_path',
        y_col='disease',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def create_fixed_data_generators(train_df, val_df, test_df):
    print("🔄 Creating fixed data generators...")
    
    # Use the SAME generator for ALL sets initially
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow_from_dataframe(
        train_df,
        x_col='file_path',
        y_col='disease',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = datagen.flow_from_dataframe(
        val_df,
        x_col='file_path', 
        y_col='disease',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = datagen.flow_from_dataframe(
        test_df,
        x_col='file_path',
        y_col='disease',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


train_gen, val_gen, test_gen = create_fixed_data_generators(train_df, val_df, test_df)


# 5. Trying different Model Architectures
def create_mobilenet_model(num_classes):
    print("🏗️ Building MobileNetV2 model...")
    
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_fixed_simple_cnn(num_classes):
    print("🏗️ Building Fixed Custom CNN model...")
    
    model = tf.keras.Sequential([
        # Explicit Input Layer
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second Convolutional Block  
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_resnet_model(num_classes):
    print("🏗️ Building ResNet50 model...")
    
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_nasnet_model(num_classes):
    print("🏗️ Building NASNetMobile model...")
    
    base_model = tf.keras.applications.NASNetMobile(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),  # Even smaller
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

num_classes = len(label_encoder.classes_)
model = create_mobilenet_model(num_classes)

# Add class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)
class_weight_dict = dict(enumerate(class_weights))
print("⚖️ Class weights:", class_weight_dict)

# Better compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("📋 Model Summary:")
model.summary()

# Debugging data pipeline
def debug_data_pipeline():
    print("🔍 Debugging data pipeline...")
    
    # Check one batch from train generator
    x_batch, y_batch = next(train_gen)
    print(f"Batch shape: {x_batch.shape}")
    print(f"Labels shape: {y_batch.shape}")
    print(f"Data range: [{x_batch.min():.3f}, {x_batch.max():.3f}]")
    print(f"Unique labels: {np.unique(np.argmax(y_batch, axis=1))}")
    
    # Check if images are loaded correctly
    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(x_batch[i])
        plt.title(f'Label: {np.argmax(y_batch[i])}')
        plt.axis('off')
    plt.show()

debug_data_pipeline()


# 6. Training with Callbacks
print("🎯 Starting training...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # Monitor accuracy, not loss
        patience=10,            # Reduced patience
        restore_best_weights=True,
        verbose=1,
        mode='max'              # Maximize accuracy
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_cattle_disease_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# Initial training
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

print("✅ Initial training completed!")

# 7. Evaluate the Model
def evaluate_model(model, test_generator, label_encoder):
    print("📊 Evaluating model on test set...")
    
    # Evaluate on test set
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
    print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"📉 Test Loss: {test_loss:.4f}")
    print(f"🎯 Test Precision: {test_precision:.4f}")
    print(f"🎯 Test Recall: {test_recall:.4f}")
    
    # Predictions
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # True labels
    y_true = test_generator.classes
    
    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n📈 Classification Report:")
    print(classification_report(y_true, y_pred_classes, 
                              target_names=label_encoder.classes_))
    
    # Confusion matrix
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
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_accuracy

test_accuracy = evaluate_model(model, test_gen, label_encoder)

# 8. Save the Model and Utilities
print("💾 Saving model and utilities...")

# Save the final model
model.save('cattle_skin_disease_model.h5')

# Save label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

print("✅ Model and utilities saved successfully!")
print(f"📁 Saved files:")
print(f"   - cattle_skin_disease_model.h5 (Model)")
print(f"   - label_encoder.pkl (Label encoder)")
print(f"   - confusion_matrix.png (Confusion matrix)")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"📊 Final Test Accuracy: {test_accuracy:.2%}")
print(f"🏷️ Classes trained: {len(label_encoder.classes_)}")