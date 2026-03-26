Multimodal Skin Cancer CDSS 🩺🔬
Clinical Decision Support System using Deep Learning & Explainable AI (XAI)
This repository contains a state-of-the-art Clinical Decision Support System (CDSS) designed to assist dermatologists in the automated classification of skin lesions. The system leverages Multimodal Data Fusion (Images + Clinical Metadata) to provide high-accuracy diagnostic suggestions.

🌟 Key Features
Multimodal AI Architecture: Combines a ResNet18 (Computer Vision) with an MLP (Tabular Data) to analyze both dermoscopic images and patient metadata (Age, Sex, Anatomical Site).

Explainable AI (Grad-CAM): Generates real-time heatmaps to visualize which morphological features the AI focused on for its decision.

Content-Based Image Retrieval (CBIR): Automatically retrieves and displays the top 3 most visually similar historical cases from a database of 2,000 confirmed biopsies.

Quality Control Module: Built-in OpenCV filters to detect and reject blurry or underexposed images before analysis.

Automated Medical Reporting: Generates a professional PDF report including patient data, AI predictions, and Grad-CAM visualizations.

🛠️ Tech Stack
Deep Learning: PyTorch, Torchvision

Frontend/UI: Streamlit

Computer Vision: OpenCV, PIL

Explainability: PyTorch-Grad-CAM

Database/Similarity: Scikit-Learn (Cosine Similarity)

Reporting: FPDF2

Hardware Acceleration: Apple Silicon (MPS) / CUDA supported

📊 Dataset
The model was trained and evaluated on the ISIC 2019 Challenge Dataset, consisting of 25,331 dermoscopic images across 9 diagnostic categories:

Melanoma (MEL)

Melanocytic nevus (NV)

Basal cell carcinoma (BCC)

Actinic keratosis (AK)

Benign keratosis (BKL)

Dermatofibroma (DF)

Vascular lesion (VASC)

Squamous cell carcinoma (SCC)

Unknown (UNK)

🚀 Installation & Usage
Clone the repository:

Bash
git clone https://github.com/YOUR_USERNAME/Skin-Cancer-CDSS.git
cd Skin-Cancer-CDSS
Install dependencies:

Bash
pip install -r requirements.txt
Run the Application:

Bash
streamlit run app.py
📝 Disclaimer
This system is a research project and is NOT intended for clinical use without expert supervision. All AI-generated diagnoses must be verified by a certified dermatologist.
