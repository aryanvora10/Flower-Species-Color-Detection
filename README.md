# 🌸 Flower Species & Accurate Color Detection Web App

> An intelligent Computer Vision and Deep Learning pipeline that classifies 5 flower species and mathematically extracts their true dominant color by isolating the foreground.

## 📖 Overview

Standard color detection algorithms often fail on images of nature because they confuse the background (like green grass, brown dirt, or a blue sky) with the object itself. 

This project solves the **Background Color Problem** by combining Deep Learning and classic Computer Vision. It uses a fine-tuned model to identify the flower, applies an advanced segmentation algorithm to "cut out" the petals from the background, and finally calculates the exact true color of the flower.

### ✨ Key Features
* **High-Accuracy Classification:** Identifies Daisy, Dandelion, Rose, Sunflower, and Tulip using a custom-trained **EfficientNetB0** model.
* **Smart Background Removal:** Utilizes **OpenCV's GrabCut** algorithm to dynamically calculate bounding boxes and isolate the flower from its surroundings.
* **Precise Color Extraction:** Applies **K-Means Clustering** strictly to the isolated foreground pixels to find the true dominant RGB values, mapping them to standard CSS color names.
* **Interactive UI:** A lightweight **Flask** web interface where users can upload images and see the original image, the segmented cutout, and the final predictions.

---

## 🧠 Model Architecture & Training

The training pipeline (`flower_pipeline.ipynb`) uses Transfer Learning to achieve maximum accuracy:

1. **Data Augmentation:** Random horizontal/vertical flips, rotations (20%), and zooming (20%) are applied to prevent overfitting.
2. **Phase 1 (Base Training):** The `EfficientNetB0` base (pre-trained on ImageNet) is frozen. A custom classification head (Global Average Pooling -> Dense 128 -> Dense 5) is trained using the Adam optimizer (Learning Rate: `1e-3`).
3. **Phase 2 (Fine-Tuning):** The top 20 layers of the EfficientNet base are unfrozen. The model is fine-tuned with a severely reduced learning rate (`1e-5`) to allow the model to specialize meticulously on the 5 flower classes without destroying the pre-trained weights. Early stopping is utilized to save the best weights (`flower_species_perfect.h5`).

---

## ⚙️ The Color Extraction Algorithm

1. **Bounding Box Calculation:** Calculates a dynamic `[w*0.1, h*0.1, w*0.8, h*0.8]` central region where the flower is most likely located.
2. **Segmentation:** Passes the image and coordinates into `cv2.grabCut` to generate a foreground mask, setting all background pixels to white (255, 255, 255).
3. **Clustering:** Extracts only the valid foreground pixels and passes them to `sklearn.cluster.KMeans` (n_clusters=3). The center of the largest cluster is returned as the dominant RGB color.

---

## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow, Keras, EfficientNetB0
* **Computer Vision & ML:** OpenCV (cv2), Scikit-Learn (K-Means)
* **Backend:** Python 3, Flask
* **Data Processing:** NumPy, Collections

---

## 🚀 Getting Started

### 1. Clone the repository:
```bash
git clone [https://github.com/YOUR_USERNAME/Flower-Species-Color-Detection.git](https://github.com/YOUR_USERNAME/Flower-Species-Color-Detection.git)
cd Flower-Species-Color-Detection
