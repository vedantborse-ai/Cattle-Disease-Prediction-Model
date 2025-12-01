import tensorflow as tf

print("Loading .h5 model...")
model = tf.keras.models.load_model("best_cattle_disease_model.h5")

print("Saving as TensorFlow SavedModel...")
model.save("saved_model", save_format="tf")

print("Conversion successful!")
