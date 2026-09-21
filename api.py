import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI(title="Crop Agent API")

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
                        resultDiv.innerHTML = "Prediction: Class " + data.prediction + " <br> Confidence: " + data.confidence.toFixed(4);
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
        
        return {
            "prediction": idx,
            "confidence": prob
        }
    except Exception as e:
        return {"error": str(e)}
