import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import pickle

# --- ΡΥΘΜΙΣΕΙΣ ---
IMG_DIR = "data/images"
LABELS_CSV = "data/ISIC_2019_Training_GroundTruth.csv"
MODEL_PATH = "models/isic2019_resnet18_multimodal.pth"
OUTPUT_DB = "data/cbir_database.pkl"
NUM_SAMPLES = 2000 # Παίρνουμε 2.000 εικόνες για να τρέξει γρήγορα το script

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("1. Φόρτωση του Μοντέλου Εικόνας...")
# Φορτώνουμε ΜΟΝΟ το Image Branch (ResNet18) από το Πολυτροπικό Μοντέλο
model = models.resnet18(weights=None)
model.fc = nn.Identity() # Αφαιρούμε το τελευταίο επίπεδο για να πάρουμε τα 512 Features

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
# Απομονώνουμε μόνο τα βάρη της εικόνας
image_branch_weights = {k.replace('image_branch.', ''): v for k, v in state_dict.items() if k.startswith('image_branch.')}
model.load_state_dict(image_branch_weights)
model = model.to(device)
model.eval()

print("2. Φόρτωση λίστας εικόνων...")
df = pd.read_csv(LABELS_CSV)
df['target'] = df[CLASS_NAMES].idxmax(axis=1)

# Παίρνουμε ένα τυχαίο δείγμα 2.000 εικόνων
sample_df = df.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(f"3. Εξαγωγή Χαρακτηριστικών (Feature Extraction) για {NUM_SAMPLES} εικόνες. Παρακαλώ περιμένετε...")
database = []

with torch.no_grad():
    for idx, row in sample_df.iterrows():
        img_name = row['image']
        label = row['target']
        img_path = os.path.join(IMG_DIR, f"{img_name}.jpg")
        
        try:
            image = Image.open(img_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)
            # Εξάγουμε το διάνυσμα 512 διαστάσεων
            features = model(img_tensor).cpu().numpy().flatten()
            
            database.append({
                'image_name': img_name,
                'label': label,
                'features': features
            })
        except Exception as e:
            continue
            
        if (idx + 1) % 200 == 0:
            print(f"--> Επεξεργάστηκαν {idx + 1}/{NUM_SAMPLES} εικόνες...")

with open(OUTPUT_DB, 'wb') as f:
    pickle.dump(database, f)

print(f"✅ Η βάση δεδομένων CBIR δημιουργήθηκε επιτυχώς στο: {OUTPUT_DB}")