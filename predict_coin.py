import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model("coin_classification_model1.h5")

# Define coin denominations
denominations = {0: '1', 1: '2', 2: '5', 3: '10', 4: '20'}

# Function to preprocess and predict
def predict_coin(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Resize the image to 640x640 while maintaining aspect ratio
    h, w = img.shape[:2]
    if h != w:
        # Pad the image to make it square
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Resize to 640x640
    img = cv2.resize(img, (640, 640))
    
    # Normalize pixel values (for model input)
    img_normalized = img / 255.0
    return img, img_normalized  # Return both original and normalized images

# Function to process multiple images
def predict_multiple_images(image_folder):
    # Get list of image paths
    image_paths = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder)]
    image_paths = [img for img in image_paths if img.endswith(('.jpg', '.jpeg', '.png'))]  # Filter valid image files

    # Preprocess all images
    images = []
    images_normalized = []
    for img_path in image_paths:
        img, img_normalized = predict_coin(img_path)
        images.append(img)  # Original image for display
        images_normalized.append(img_normalized)  # Normalized image for prediction

    images_normalized = np.array(images_normalized)  # Convert list of normalized images to a numpy array

    # Make predictions for all images
    predictions = model.predict(images_normalized)
    predicted_classes = np.argmax(predictions, axis=1)

    # Display results with original images
    for img_path, img, pred_class in zip(image_paths, images, predicted_classes):
        # Display the original image with the predicted denomination
        cv2.putText(img, f"Denomination: {denominations[pred_class]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Prediction", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
image_folder = "C:/Users/HP/Desktop/coinvision/TEST"  # Use raw string to avoid escape sequence issues
predict_multiple_images(image_folder)