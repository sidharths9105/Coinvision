import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

# Load trained model
MODEL_PATH = "coin_classifier_finetuned3.h5"
model = load_model(MODEL_PATH)

# Define class labels and their values
class_labels = {0: "1 Rupee", 1: "10 Rupees", 2: "2 Rupees", 3: "5 Rupees"}
coin_values = {"1 Rupee": 1, "10 Rupees": 10, "2 Rupees": 2, "5 Rupees": 5}

# GUI Setup
root = tk.Tk()
root.title("COINVISION")
root.state('zoomed')  # Set window to full screen
root.resizable(False, False)  # Fix window size

# Layout Frames
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

image_frame = tk.Frame(root)
image_frame.pack()

result_frame = tk.Frame(root)
result_frame.pack(pady=10)

def process_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 2)

    # Morphological transformation to close gaps
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Hough Circle Transform for coin detection
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
                               param1=50, param2=30, minRadius=30, maxRadius=100)
    
    total_value = 0  # Store total coin value
    coin_counts = {"1 Rupee": 0, "10 Rupees": 0, "2 Rupees": 0, "5 Rupees": 0}
    
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        circles = sorted(circles, key=lambda x: x[2], reverse=True)[:4]  # Take top 4

        for i in circles:
            x, y, r = i[0], i[1], i[2]
            adjusted_radius = int(r * 1.2)
            x1, y1 = max(0, x - adjusted_radius), max(0, y - adjusted_radius)
            x2, y2 = min(image.shape[1], x + adjusted_radius), min(image.shape[0], y + adjusted_radius)
            roi = image[y1:y2, x1:x2]

            # Preprocess ROI
            roi_resized = cv2.resize(roi, (224, 224))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            roi_normalized = roi_rgb / 255.0  
            roi_expanded = np.expand_dims(roi_normalized, axis=0)

            # Predict coin type
            predictions = model.predict(roi_expanded)
            predicted_class = np.argmax(predictions)
            coin_label = class_labels[predicted_class]
            total_value += coin_values[coin_label]
            coin_counts[coin_label] += 1

            # Draw circle and label
            cv2.circle(image, (x, y), adjusted_radius, (0, 255, 0), 2)
            cv2.putText(image, coin_label, (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (255, 0, 0), 2)
    
    # Resize image for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    img_pil = img_pil.resize((800, 600), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Update GUI image display
    panel.config(image=img_tk)
    panel.image = img_tk
    
    # Update coin count and total value display
    count_text = "\n".join([f"{key}: {value}" for key, value in coin_counts.items()])
    count_label.config(text=f"Coin Count:\n{count_text}")
    result_label.config(text=f"Total Value: {total_value} Rupees")

# GUI Layout
btn_upload = tk.Button(top_frame, text="Upload Image", command=process_image, font=("Arial", 14))
btn_upload.pack()

panel = tk.Label(image_frame)
panel.pack()

count_label = tk.Label(result_frame, text="Coin Count:\n1 Rupee: 0\n2 Rupees: 0\n5 Rupees: 0\n10 Rupees: 0", font=("Arial", 14), justify="left")
count_label.pack()

result_label = tk.Label(result_frame, text="Total Value: 0 Rupees", font=("Arial", 16, "bold"))
result_label.pack()

root.mainloop()
