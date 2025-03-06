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

# ✅ Load Alzheimer's model **only once** at startup
alzheimers_model = None

@app.on_event("startup")
def load_alzheimers_model():
    global alzheimers_model
    alzheimers_model = tf.keras.models.load_model("alzheimers_model.h5")

alzheimers_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Preprocess function
def preprocess_alzheimers_image(image):
    img = Image.open(image).resize((224, 224))
    img_array = np.asarray(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict_alzheimers(image):
    img_array = preprocess_alzheimers_image(image)
    predictions = alzheimers_model.predict(img_array)
    predicted_class = alzheimers_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    gc.collect()  # Free up memory
    return predicted_class, confidence

# FastAPI Endpoint
@app.post("/predict_alzheimers/")
async def predict_alzheimers_endpoint(file: UploadFile = File(...)):
    prediction, confidence = predict_alzheimers(file.file)
    return {"diagnosis": prediction, "confidence": f"{confidence:.2f}%"}
