import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader


API_KEY = "15374101225" 
api_key_header = APIKeyHeader(name="X-API-Key")
IMAGE_SIZE = 180
MODEL_CANDIDATES = [
    "crop_simple.keras",
    "crop_multiclass.keras",
    "crop_disease.keras",
    "crop_disease_model.keras",
]

# Global dictionary to hold the model so it persists across requests
app_data = {}



def get_model_path():
    for name in MODEL_CANDIDATES:
        if Path(name).exists():
            return name
    return None


def get_class_names():
    train_dir = Path("datasets/classify_multiclass/train")
    if train_dir.exists():
        names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
        if names:
            return names

    return ["Healthy", "Corn Blight", "Tomato Leaf Curl"]



def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key. Access Denied.")
    return api_key



# ensures the model is loaded ONCE when the server boots
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading TensorFlow model...")
    model_path = get_model_path()
    if model_path is None:
        print("No model file found. Please train a model first.")
        app_data["model"] = None
    else:
        app_data["model"] = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully: {model_path}")
    yield
    
    app_data.clear()

app = FastAPI(title="Crop Vision API", lifespan=lifespan)


# PREDICTION ENDPOINT 
@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict_image(file: UploadFile = File(...)):
    model = app_data.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train a model first.")

    #  Read the image byte data
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
   
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    #  Convert image to numpy array 
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Expand dimensions to create a batch of 1: shape becomes (1, 180, 180, 3)
    img_tensor = np.expand_dims(img_array, axis=0)

    #  Run the model prediction
    predictions = model.predict(img_tensor, verbose=0)
    predicted_class_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    class_names = get_class_names()
    if predicted_class_index < len(class_names):
        predicted_label = class_names[predicted_class_index]
    else:
        predicted_label = str(predicted_class_index)

    return {
        "status": "success",
        "predicted_class": predicted_label,
        "confidence": confidence,
        "class_index": predicted_class_index,
    }