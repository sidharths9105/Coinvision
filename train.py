import os
import numpy as np
import cv2
from sklearn.utils import resample
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Oversample Minority Classes
def oversample_minority_classes(image_paths, labels, target_size):
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    max_count = max(class_counts)

    resampled_image_paths = []
    resampled_labels = []
    for class_label in unique_classes:
        class_image_paths = [path for path, label in zip(image_paths, labels) if label == class_label]
        if len(class_image_paths) < max_count:
            class_image_paths = resample(class_image_paths, replace=True, n_samples=max_count, random_state=42)
        resampled_image_paths.extend(class_image_paths)
        resampled_labels.extend([class_label] * len(class_image_paths))

    return resampled_image_paths, resampled_labels

# Step 2: Load and Preprocess Images
def load_and_preprocess_images(image_paths, target_size=(640,640 )):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, target_size)  # Resize to target size
        img = img / 255.0  # Normalize pixel values to [0, 1]
        images.append(img)
    return np.array(images)

# Step 3: Prepare Dataset
input_folder = "C:/Users/HP/Desktop/coinvision/dataset"  # Replace with your dataset path
class_folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]  # Only directories
image_paths = []
labels = []

# Load image paths and labels
for class_label, class_folder in enumerate(class_folders):
    class_path = os.path.join(input_folder, class_folder)
    if os.path.isdir(class_path):  # Ensure it's a directory
        class_images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg','.JPG', '.jpeg', '.png'))]
        image_paths.extend(class_images)
        labels.extend([class_label] * len(class_images))

# Oversample minority classes
resampled_image_paths, resampled_labels = oversample_minority_classes(image_paths, labels, target_size=300)

# Load and preprocess the oversampled images
X_train = load_and_preprocess_images(resampled_image_paths, target_size=(640, 640))
y_train = to_categorical(resampled_labels, num_classes=len(class_folders))

# Step 4: Visualize a Preprocessed Image and Its Label
def visualize_preprocessed_image(image, label, class_names):
    plt.imshow(image)
    plt.title(f"Label: {class_names[label]}")
    plt.axis('off')
    plt.show()

# Define class names
class_names = {0: '1 Rupee', 1: '2 Rupees', 2: '5 Rupees', 3: '10 Rupees'}  # Replace with your class names

# Visualize the first preprocessed image and its label
visualize_preprocessed_image(X_train[0], np.argmax(y_train[0]), class_names)

# Step 5: Build the Model
base_model = MobileNetV2(input_shape=(640, 640, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_folders), activation='softmax')  # Output layer
])

# Step 6: Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Step 8: Plot Training/Validation Accuracy and Loss
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# Step 9: Save the Model
model.save("coin_classification_mobilenetv2.h5")
print("Model saved as coin_classification_mobilenetv2.h5")