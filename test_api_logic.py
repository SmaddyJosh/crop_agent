import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

def prepare_image(image_bytes, target_size=(180, 180)):
    img = Image.open(BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)

model = tf.keras.models.load_model('crop_simple.keras')
with open('datasets/all_data/Healthy/IMG_0216_JPG.rf.2f2281def8c9e88325cf0bcfb3689634.jpg', 'rb') as f:
    img_bytes = f.read()

img = prepare_image(img_bytes)
preds = model.predict(img)
idx = int(np.argmax(preds[0]))
prob = float(np.max(preds[0]))

print(f"Predicted index: {idx} with confidence {prob}")
