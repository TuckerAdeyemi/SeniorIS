import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import io
from pathlib import Path

# Load class names
BASE_DIR = Path(__file__).parent
with open(BASE_DIR / 'class_names.json', 'r') as f:
    class_names = json.load(f)

# Load model
MODEL_PATH = BASE_DIR / 'galaxy_classifier.keras'
model = load_model(MODEL_PATH)

IMG_SIZE = (126, 126)

def predict_image(file_storage):
    """Predict galaxy classification from uploaded image"""
    # Read file and convert to image
    image_bytes = file_storage.read()
    image_stream = io.BytesIO(image_bytes)
    
    # Load and preprocess image to match training size
    img = load_img(image_stream, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, 0) 
    
    # Predict
    probs = model.predict(arr, verbose=0)[0]
    
    # Get top prediction
    top_idx = np.argsort(probs)[::-1]
    pred_idx = top_idx[0]
    pred_class = class_names[pred_idx]
    conf = float(probs[pred_idx])
    
    # Get top 3
    top3 = [(class_names[i], float(probs[i])) for i in top_idx[:3]]
    
    return {
        'classification': pred_class,
        'probability': conf,
        'top_3': top3
    }