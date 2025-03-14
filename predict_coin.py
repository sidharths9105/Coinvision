import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import os
# Load the model
model = load_model('coin_classification_model.h5')
IMG_SIZE = 224
labels = sorted(os.listdir('C:/Users/HP/Desktop/coinvision/dataset'))



def predict_coin(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read {image_path}")
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    pred = model.predict(img)
    predicted_label = labels[np.argmax(pred)]  # Get the class with highest probability
    return predicted_label

print(predict_coin('test01.jpg'))
