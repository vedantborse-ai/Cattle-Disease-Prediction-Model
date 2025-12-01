import tensorflow as tf

# Load your existing .h5 model
model = tf.keras.models.load_model("best_cattle_disease_model.h5")

# Export as a SavedModel folder (portable across versions)
model.export("best_cattle_disease_model")
