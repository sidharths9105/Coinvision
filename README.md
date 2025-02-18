# CoinVision - Coin Detection and Counting System

## 📌 Project Overview
CoinVision is a machine learning-based system designed to detect and count coins from images. The project aims to use image processing techniques and deep learning (CNN) to identify different Indian coins and calculate their total value.

## 🚀 Current Progress
### ✅ Completed:
- 📂 **Dataset Collection**  
  - Downloaded datasets containing various Indian coins.
  - Selected one primary dataset for further processing.

- 🏗 **Folder Structure Setup**  
  - Organized dataset into `/dataset/raw/rupee_1/` and other relevant directories.

- 🏷 **Image Labeling**  
  - Labeled all images with the class name `coin` and assigned corresponding values.

- ✂ **Image Preprocessing**  
  - Resized images to maintain a uniform size for CNN training.  
  - Explored cropping methods to preserve clarity without losing coin features.  
  - Decided not to classify front and back sides separately, as the goal is only to detect and count.

### ⏳ In Progress:
- 🔄 **Finalizing Image Preprocessing Pipeline**  
  - Exploring ways to crop images while keeping clarity intact.  
  - Deciding on a suitable input resolution for CNN training.  

- 🧠 **Model Selection & Training**  
  - Choosing between CNN architectures (e.g., MobileNet, ResNet).  
  - Preparing data augmentation techniques to improve model robustness.  

### ⏭️ Next Steps:
- 🔢 **Training the Model** on processed datasets.  
- 🎯 **Testing Accuracy & Fine-Tuning** hyperparameters.  
- 🖥 **Deploying as an App/Web Interface** (Future Goal).  
