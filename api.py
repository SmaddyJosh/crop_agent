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
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Crop AI Agent</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-color: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --primary: #3b82f6;
                --primary-hover: #2563eb;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --border: rgba(255, 255, 255, 0.1);
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                color: var(--text-main);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 2rem;
            }
            .app-container {
                display: grid;
                grid-template-columns: 1fr 1.2fr;
                gap: 2rem;
                max-width: 1200px;
                width: 100%;
            }
            .card {
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 20px 40px rgba(0,0,0,0.4);
                display: flex;
                flex-direction: column;
            }
            h2 { font-weight: 600; margin-bottom: 1rem; color: #fff; }
            p { color: var(--text-muted); line-height: 1.6; margin-bottom: 1.5rem; }
            
            /* Upload Area */
            .upload-area {
                border: 2px dashed var(--primary);
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                position: relative;
                background: rgba(59, 130, 246, 0.05);
            }
            .upload-area:hover { background: rgba(59, 130, 246, 0.1); }
            .upload-area input {
                position: absolute; width: 100%; height: 100%; top: 0; left: 0;
                opacity: 0; cursor: pointer;
            }
            .preview-container {
                margin-top: 1.5rem;
                display: none;
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid var(--border);
            }
            #preview { width: 100%; display: block; }
            
            button {
                background: var(--primary);
                color: white;
                border: none;
                padding: 1rem;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                margin-top: 1.5rem;
                width: 100%;
            }
            button:hover { background: var(--primary-hover); transform: translateY(-2px); }
            button:disabled { background: #475569; cursor: not-allowed; transform: none; }
            
            /* Chat Area */
            .chat-container {
                display: flex;
                flex-direction: column;
                height: 100%;
                min-height: 400px;
            }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding-right: 1rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
                margin-bottom: 1rem;
            }
            .message {
                padding: 1rem;
                border-radius: 12px;
                max-width: 85%;
                animation: fadeIn 0.3s ease;
                line-height: 1.5;
            }
            .message.system {
                background: rgba(255, 255, 255, 0.05);
                align-self: flex-start;
                border: 1px solid var(--border);
            }
            .message.agent {
                background: linear-gradient(135deg, #3b82f6, #6366f1);
                align-self: flex-start;
                color: white;
            }
            .message.user {
                background: rgba(255,255,255,0.1);
                align-self: flex-end;
                border: 1px solid var(--border);
            }
            .chat-input-area {
                display: flex;
                gap: 0.5rem;
                margin-top: auto;
            }
            input[type="text"] {
                flex: 1;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid var(--border);
                background: rgba(0,0,0,0.2);
                color: white;
                font-family: inherit;
            }
            input[type="text"]:focus { outline: none; border-color: var(--primary); }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .loading {
                display: inline-block;
                width: 20px; height: 20px;
                border: 3px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 1s ease-in-out infinite;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
            
            @media (max-width: 768px) {
                .app-container { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="app-container">
            <!-- Left: Upload Card -->
            <div class="card">
                <h2>Crop Analysis</h2>
                <p>Upload a photo of a leaf to identify diseases and get treatment recommendations.</p>
                
                <div class="upload-area">
                    <input type="file" id="imageInput" accept="image/*" />
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 1rem;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                    <div style="font-weight: 600;">Click or drag image here</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.5rem;">PNG, JPG up to 10MB</div>
                </div>
                
                <div class="preview-container" id="previewContainer">
                    <img id="preview" src="#" alt="Preview" />
                </div>
                
                <button id="analyzeBtn" onclick="predict()" disabled>Analyze Crop</button>
            </div>
    
            <!-- Right: Chat Card -->
            <div class="card">
                <h2>AI Agent Assistant</h2>
                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="message system">
                            Hello! I am your Crop AI Assistant. Please upload a photo on the left, and I will analyze it for diseases and give you treatment advice.
                        </div>
                    </div>
                    <div class="chat-input-area">
                        <input type="text" id="chatInput" placeholder="Ask a follow-up question (Coming soon...)" disabled />
                        <button style="width: auto; margin-top: 0;" disabled>Send</button>
                    </div>
                </div>
            </div>
        </div>
    
        <script>
            const input = document.getElementById('imageInput');
            const preview = document.getElementById('preview');
            const previewContainer = document.getElementById('previewContainer');
            const chatMessages = document.getElementById('chatMessages');
            const analyzeBtn = document.getElementById('analyzeBtn');
    
            function addMessage(text, type) {
                const msg = document.createElement('div');
                msg.className = `message ${type}`;
                msg.innerHTML = text;
                chatMessages.appendChild(msg);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
    
            input.onchange = evt => {
                const [file] = input.files;
                if (file) {
                    preview.src = URL.createObjectURL(file);
                    previewContainer.style.display = 'block';
                    analyzeBtn.disabled = false;
                }
            }
    
            async function predict() {
                if (!input.files[0]) return;
                
                const formData = new FormData();
                formData.append("file", input.files[0]);
    
                addMessage("Analyzing image...", "user");
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<div class="loading"></div>';
    
                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (data.error) {
                        addMessage("Error: " + data.error, "system");
                    } else {
                        let conf = (data.confidence * 100).toFixed(1) + "%";
                        // Display Agent Response
                        if (data.agent_response) {
                            addMessage(`<b>Prediction:</b> ${data.prediction} (${conf})<br><br>${data.agent_response}`, "agent");
                        } else {
                            addMessage(`<b>Prediction:</b> ${data.prediction} (${conf})`, "agent");
                        }
                    }
                } catch (e) {
                    addMessage("Error connecting to the API.", "system");
                } finally {
                    analyzeBtn.disabled = false;
                    analyzeBtn.innerHTML = 'Analyze Crop';
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
