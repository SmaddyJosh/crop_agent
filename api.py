import json
import os
from io import BytesIO
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

app = FastAPI(title="Crop Agent API")

def get_class_names():
    train_dir = Path("datasets/classify_multiclass/train")
    if train_dir.exists():
        names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
        if names:
            return names
    return ["Healthy", "Corn Blight", "Tomato Leaf Curl"]

# Load model globally when API starts
MODEL_PATH = 'crop_simple.keras'
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None
    print(f"Warning: Model not found at {MODEL_PATH}")

def prepare_image(image_bytes, target_size=(180, 180)):
    # Load image from bytes
    img = Image.open(BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)

@app.get("/", response_class=HTMLResponse)
async def home():
    classes = get_class_names()
    classes_list = "".join(f"<li>{c}</li>" for c in classes)
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Crop Model</title>
        <style>
            body { font-family: sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; text-align: center; }
            #preview { max-width: 300px; margin-top: 20px; }
            .result { margin-top: 20px; font-weight: bold; font-size: 1.2em; }
            .classes-list { text-align: left; margin-top: 30px; background: #f5f5f5; padding: 20px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Upload an image to test the model</h2>
            <input type="file" id="imageInput" accept="image/*" />
            <br>
            <img id="preview" src="#" alt="Image preview" style="display: none;" />
            <br><br>
            <button onclick="predict()">Predict</button>
            <div class="result" id="result"></div>
            
            <div class="classes-list">
                <h3>Supported Classes:</h3>
                <ul>
                    """ + classes_list + """
                </ul>
            </div>
        </div>

        <script>
            const input = document.getElementById('imageInput');
            const preview = document.getElementById('preview');
            const resultDiv = document.getElementById('result');

            input.onchange = evt => {
                const [file] = input.files;
                if (file) {
                    preview.src = URL.createObjectURL(file);
                    preview.style.display = 'block';
                    resultDiv.innerHTML = '';
                }
            }

            async function predict() {
                if (!input.files[0]) {
                    alert("Please select an image first!");
                    return;
                }
                const formData = new FormData();
                formData.append("file", input.files[0]);

                resultDiv.innerHTML = "Predicting...";
                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    });
                    const data = await response.json();
                    if (data.error) {
                        resultDiv.innerHTML = "Error: " + data.error;
                    } else {
                        let html = "Prediction: Class " + data.prediction + " <br> Confidence: " + data.confidence.toFixed(4);
                        if (data.agent_response) {
                            html += "<br><br><b>Agent says:</b><br>" + data.agent_response;
                        }
                        resultDiv.innerHTML = html;
                    }
                } catch (e) {
                    resultDiv.innerHTML = "Error connecting to the API.";
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded on the server."}
    
    try:
        contents = await file.read()
        img = prepare_image(contents)
        
        preds = model.predict(img)
        idx = int(np.argmax(preds[0]))
        prob = float(np.max(preds[0]))
        
        class_names = get_class_names()
        class_name = class_names[idx] if idx < len(class_names) else str(idx)
        
        # Load knowledge base
        knowledge = {}
        kb_path = Path("knowledge_base.json")
        if kb_path.exists():
            with open(kb_path, "r") as f:
                knowledge = json.load(f)
                
        crop_info = knowledge.get(class_name, {})
        description = crop_info.get("description", "No detailed description available.")
        recommendation = crop_info.get("recommendation", "No specific recommendations available.")

        if prob < 0.7:
            agent_response = (
                f"I'm a bit uncertain (confidence: {prob:.2f}), but this looks like it could be {class_name}. "
                f"If it is {class_name}: {description} "
                f"Recommendation: {recommendation}"
            )
        else:
            agent_response = (
                f"I am highly confident ({prob:.2f}) this is {class_name}. "
                f"{description} "
                f"Recommendation: {recommendation}"
            )
        
        return {
            "prediction": class_name,
            "confidence": prob,
            "agent_response": agent_response
        }
    except Exception as e:
        return {"error": str(e)}
