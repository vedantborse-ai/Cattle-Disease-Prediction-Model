import tensorflow as tf

model = tf.keras.models.load_model("best_cattle_disease_model.h5")
model.export("exported_model")