import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import gc

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Label mappings for Alzheimer's classification
alzheimers_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Define preprocessing function
def preprocess_alzheimers_image(image):
    """ Preprocess image for Alzheimer's model using MobileNetV2 preprocessing. """
    img = Image.open(image).convert("RGB").resize((224, 224))  # Convert to RGB and resize
    img_array = np.asarray(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    return img_array

# Prediction function
def predict_alzheimers(image):
    """ Load model dynamically, predict, then unload to save memory. """
    model = tf.keras.models.load_model("alzheimers_model.h5")  # Load model on demand
    img_array = preprocess_alzheimers_image(image)
    predictions = model.predict(img_array)
    
    predicted_class = alzheimers_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    del model  # Delete model from memory
    gc.collect()  # Free up memory

    return predicted_class, confidence

# FastAPI Endpoint
@app.post("/predict_alzheimers/")
async def predict_alzheimers_endpoint(file: UploadFile = File(...)):
    """ Endpoint to receive an image, process it, and return the prediction. """
    prediction, confidence = predict_alzheimers(file.file)
    return {"diagnosis": prediction, "confidence": f"{confidence:.2f}%"}

