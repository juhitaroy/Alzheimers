import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
import shutil
import os

app = FastAPI()

# Load Alzheimer's model
alzheimers_model = tf.keras.models.load_model("alzheimers_model.h5")

# Define preprocessing function
def preprocess_alzheimers_image(image_path):
    """ Preprocess image for Alzheimer's model using MobileNetV2 preprocessing. """
    img = Image.open(image_path).resize((224, 224))  # Fixed to (224, 224)
    img_array = np.asarray(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    return img_array

# Define label mappings
alzheimers_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Prediction function
def predict_alzheimers(image_path):
    img_array = preprocess_alzheimers_image(image_path)
    predictions = alzheimers_model.predict(img_array)
    predicted_class = alzheimers_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

# FastAPI Endpoint
@app.post("/predict_alzheimers/")
async def predict_alzheimers_endpoint(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = predict_alzheimers(file_path)
    os.remove(file_path)  # Cleanup

    return {"diagnosis": prediction, "confidence": f"{confidence:.2f}%"}
