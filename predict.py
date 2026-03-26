import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. ΡΥΘΜΙΣΕΙΣ ---
MODEL_PATH = "models/isic2019_resnet18.pth"
# Θα σώζουμε τον θερμικό χάρτη σε αυτόν τον φάκελο
GRADCAM_OUTPUT_DIR = "data/gradcam_results" 
CLASS_NAMES = [
    'MEL (Μελάνωμα)', 'NV (Σπίλος/Ελιά)', 'BCC (Βασικοκυτταρικό Καρκίνωμα)', 
    'AK (Ακτινική Κεράτωση)', 'BKL (Καλοήθης Κεράτωση)', 'DF (Δερματοΐνωμα)', 
    'VASC (Αγγειακή Βλάβη)', 'SCC (Ακανθοκυτταρικό Καρκίνωμα)', 'UNK (Άγνωστο)'
]

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# --- 2. ΕΛΕΓΧΟΣ ΠΟΙΟΤΗΤΑΣ ---
def check_image_quality(image_path, blur_threshold=20.0, dark_threshold=40.0):
    img = cv2.imread(image_path)
    if img is None:
        return False, "Σφάλμα: Το αρχείο δεν βρέθηκε."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_value < blur_threshold:
        return False, f"ΑΠΟΡΡΙΦΘΗΚΕ: Πολύ θολή ({blur_value:.1f} < {blur_threshold})."

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness_value = np.mean(hsv[:, :, 2])
    if brightness_value < dark_threshold:
        return False, f"ΑΠΟΡΡΙΦΘΗΚΕ: Πολύ σκοτεινή ({brightness_value:.1f} < {dark_threshold})."

    return True, f"ΕΓΚΡΙΘΗΚΕ: Ποιότητα ΟΚ (Blur: {blur_value:.1f}, Brightness: {brightness_value:.1f})."


# --- 3. ΦΟΡΤΩΣΗ ΜΟΝΤΕΛΟΥ ---
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    if not os.path.exists(MODEL_PATH):
        print(f"Σφάλμα: Δεν βρέθηκε το {MODEL_PATH}.")
        sys.exit(1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval() 
    return model


# --- 4. ΠΡΟΒΛΕΨΗ ---
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0) * 100
        
        print("\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ ΔΙΑΓΝΩΣΗΣ ---")
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        for i in range(3):
            label = CLASS_NAMES[top3_idx[i]]
            prob = top3_prob[i].item()
            print(f"{i+1}. {label}: {prob:.2f}%")
            
        # Επιστρέφουμε την επικρατέστερη κλάση (top1) για να την δώσουμε στο Grad-CAM
        return top3_idx[0].item(), image_tensor
            
    except Exception as e:
        print(f"Σφάλμα κατά την ανάλυση: {e}")
        return None, None

# --- 5. ΔΗΜΙΟΥΡΓΙΑ ΘΕΡΜΙΚΟΥ ΧΑΡΤΗ (Grad-CAM) ---
def generate_gradcam(image_path, model, target_class_index, input_tensor):
    """
    Δημιουργεί και αποθηκεύει τον θερμικό χάρτη Grad-CAM.
    """
    print(f"\n[3/3] Δημιουργία Θερμικού Χάρτη (Grad-CAM) για την κλάση: {CLASS_NAMES[target_class_index]}...")
    print("-" * 50)

    # ΛΥΣΗ ΣΦΑΛΜΑΤΟΣ Μ4: Μεταφέρουμε τα πάντα στην CPU ΠΡΙΝ κάνουμε οτιδήποτε άλλο
    model = model.to('cpu')
    input_tensor = input_tensor.to('cpu')

    # 1. Ορίζουμε το επίπεδο-στόχο (το τελευταίο συνελικτικό επίπεδο του ResNet18)
    target_layers = [model.layer4[-1]]

    # 2. Αρχικοποιούμε το Grad-CAM τώρα που το μοντέλο είναι ασφαλώς στην CPU
    cam = GradCAM(model=model, target_layers=target_layers)

    # 3. Ορίζουμε τον στόχο μας (την κλάση που θέλουμε να επεξηγήσουμε)
    targets = [ClassifierOutputTarget(target_class_index)]

    # 4. Παράγουμε τον χάρτη Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # 5. Φορτώνουμε την αρχική εικόνα για το overlay (cv2 format, normalized 0-1)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255.0 # Κανονικοποίηση 0-1

    # 6. Συνδυάζουμε την αρχική εικόνα με τον θερμικό χάρτη
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    # 7. Αποθηκεύουμε το αποτέλεσμα
    os.makedirs(GRADCAM_OUTPUT_DIR, exist_ok=True)
    
    file_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(GRADCAM_OUTPUT_DIR, f"{file_basename}_gradcam.jpg")
    
    # Μετατρέπουμε πίσω σε BGR για το OpenCV save
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_filename, visualization_bgr)
    
    print(f"✅ Ο θερμικός χάρτης αποθηκεύτηκε στο: {output_filename}")
    
    # Επαναφέρουμε το μοντέλο στο αρχικό device (MPS) για να είναι έτοιμο αν τρέξουμε κάτι άλλο
    model = model.to(device)

# --- 6. ΕΚΤΕΛΕΣΗ (Main) ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        
        print(f"\n[1/3] Έλεγχος Ποιότητας: {img_path}")
        print("-" * 50)
        is_valid, message = check_image_quality(img_path)
        print(message)
        if not is_valid:
            print("Διαδικασία διεκόπη.")
            sys.exit(0)
            
        print(f"\n[2/3] Φόρτωση Μοντέλου & Πρόβλεψη...")
        print("-" * 50)
        model = load_model()
        # Παίρνουμε το top1 class index και το input tensor
        top1_idx, input_tensor = predict_image(img_path, model)
        
        if top1_idx is not None:
            # Δημιουργούμε το Grad-CAM!
            generate_gradcam(img_path, model, top1_idx, input_tensor)
        
    else:
        print("Παρακαλώ δώστε το όνομα μιας εικόνας.")