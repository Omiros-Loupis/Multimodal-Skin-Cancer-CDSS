# DermAI 🔬 - Multimodal Clinical Decision Support System (CDSS) for Skin Lesions

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

**DermAI** is an end-to-end multimodal Clinical Decision Support System (CDSS) designed to assist dermatologists in classifying skin lesions. Developed as part of a B.Sc. Thesis in Applied Informatics at the University of Macedonia (PAMAK).

This project goes beyond simple image classification by combining **Deep Learning (ResNet18)** with **Clinical Metadata**, implementing **Explainable AI (Grad-CAM)**, and featuring a **Content-Based Image Retrieval (CBIR)** system, all wrapped in a modern web interface.

## 🌟 Key Features

* **Multimodal Classification:** Fuses image data with patient clinical metadata (Age, Sex, Anatomical Site) to predict 9 different classes of skin lesions based on the ISIC 2019 dataset.
* **Explainable AI (XAI):** Utilizes Grad-CAM to generate heatmaps, showing doctors exactly which parts of the lesion the neural network focused on to make its prediction.
* **Content-Based Image Retrieval (CBIR):** Extracts feature vectors (512-dim) using ResNet18 to find and display the top 3 most visually similar verified cases from the database.
* **Automated Medical Reporting:** Generates professional, downloadable PDF reports containing patient data, predictions, heatmaps, and medical disclaimers.
* **Apple Silicon Optimized:** Native support for Apple M-series chips (MPS) for accelerated training and inference.

## 📸 Interface Showcase

| Dashboard & Prediction | Explainability (Grad-CAM) & CBIR |
| :---: | :---: |
| <img src="screenshot_1.png" width="400"> | <img src="screenshot_2.png" width="400"> |

## 🏗️ Architecture & Methodologies
* **Vision Model:** `ResNet18` (Pre-trained on ImageNet, fine-tuned).
* **Data Fusion:** Late fusion of CNN feature maps with One-Hot Encoded / Scaled clinical metadata.
* **UI Framework:** `Streamlit`.
* **PDF Generation:** `ReportLab`.
* **Preprocessing:** Implements "DullRazor" logic for digital hair removal via OpenCV.

## 📊 Dataset & Evaluation

The model was trained on the **ISIC 2019 Dataset**. Due to the heavy class imbalance of the dataset (majority of images being Nevi - NV and Melanomas - MEL) and hardware constraints limiting epochs (proof-of-concept training), the overall accuracy sits at ~53%. 

*Note: The primary goal of this system is architectural completeness (UI, XAI, CBIR, Reporting) as a foundation for a CDSS, rather than achieving state-of-the-art diagnostic accuracy.*

| Image-only Confusion Matrix | Multimodal Confusion Matrix |
| :---: | :---: |
| <img src="confusion_matrix.png" width="400"> | <img src="confusion_matrix_multimodal.png" width="400"> |

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)[ΤΟ_USERNAME_ΣΟΥ]/DermAI-Skin-Lesion-CDSS.git
cd DermAI-Skin-Lesion-CDSS
