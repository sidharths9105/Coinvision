import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Load the trained model
model = load_model("coin_classification_model.h5")

# Define coin denominations
denominations = {0: '1', 1: '2', 2: '5', 3: '10', 4: '20'}

# Function to preprocess and predict
def predict_coin(image_path):
    # Load and resize the image to 512x512 (or the size used during training)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512)) 
    plt.imshow(img)
    plt.show()
    img = img / 255.0  # Normalize pixel values
    return img

# Function to process multiple images
def predict_multiple_images(image_folder):
    # Get list of image paths
    image_paths = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder)]
    image_paths = [img for img in image_paths if img.endswith(('.jpg', '.jpeg', '.png'))]  # Filter valid image files

    # Preprocess all images
    images = [predict_coin(img_path) for img_path in image_paths]
    images = np.array(images)  # Convert list of images to a numpy array

    # Make predictions for all images
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    # Display results
    for img_path, pred_class in zip(image_paths, predicted_classes):
        print(f"Image: {os.path.basename(img_path)} | Predicted Denomination: {denominations[pred_class]}")

# Example usage
image_folder = "C:/Users/HP/Desktop/coinvision/TEST"  # Replace with the path to your folder of images
predict_multiple_images(image_folder)