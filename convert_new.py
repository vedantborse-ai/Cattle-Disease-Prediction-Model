from keras.models import load_model

print("Loading .h5 model using Keras 3...")
model = load_model("best_cattle_disease_model.h5")

print("Saving as .keras format...")
model.save("best_cattle_disease_model.keras")

print("DONE — Upload .keras file to Streamlit!")
