import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- 1. ΡΥΘΜΙΣΕΙΣ ---
IMG_DIR = "data/images"
LABELS_CSV = "data/ISIC_2019_Training_GroundTruth.csv"
META_CSV = "data/ISIC_2019_Training_Metadata.csv"
MODEL_SAVE_PATH = "models/isic2019_resnet18_multimodal.pth" # Νέο όνομα για το νέο μοντέλο

BATCH_SIZE = 32
EPOCHS = 5 # 5 Εποχές για γρήγορη εκπαίδευση
LEARNING_RATE = 0.001

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Χρήση Apple M4 GPU (MPS)!")
else:
    device = torch.device("cpu")

# --- 2. ΚΑΘΑΡΙΣΜΟΣ ΤΡΙΧΩΝ (Hair Removal - DullRazor) ---
class HairRemovalTransform(object):
    def __call__(self, img):
        cv_img = np.array(img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        grayScale = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        img_inpainted = cv2.inpaint(cv_img, mask, 1, cv2.INPAINT_TELEA)
        img_inpainted = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_inpainted)

# --- 3. ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ & ΑΠΟΦΥΓΗ DATA LEAKAGE ---
print("Φόρτωση και προετοιμασία δεδομένων...")
labels_df = pd.read_csv(LABELS_CSV)
meta_df = pd.read_csv(META_CSV)
df = pd.merge(labels_df, meta_df, on='image')

df['target'] = df[CLASS_NAMES].idxmax(axis=1)
df['label'] = df['target'].apply(lambda x: CLASS_NAMES.index(x))
df['lesion_id'] = df['lesion_id'].fillna('unknown_lesion')

# Γεμίζουμε τα κενά
df['age_approx'] = df['age_approx'].fillna(df['age_approx'].mean())
df['sex'] = df['sex'].fillna('unknown')
df['anatom_site_general'] = df['anatom_site_general'].fillna('unknown')

# 1. ΔΙΑΧΩΡΙΣΜΟΣ ΠΡΩΤΑ (Για αποφυγή Data Leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df['lesion_id']))
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

# 2. FIT ΜΟΝΟ ΣΤΟ TRAIN SET
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(train_df[['sex', 'anatom_site_general']])

scaler = StandardScaler()
scaler.fit(train_df[['age_approx']])

# 3. ΕΦΑΡΜΟΓΗ (TRANSFORM) ΣΕ TRAIN ΚΑΙ VAL
def create_clinical_features(dataframe):
    cat_features = encoder.transform(dataframe[['sex', 'anatom_site_general']])
    num_features = scaler.transform(dataframe[['age_approx']])
    return np.concatenate([num_features, cat_features], axis=1).tolist()

train_df['clinical_features'] = create_clinical_features(train_df)
val_df['clinical_features'] = create_clinical_features(val_df)

CLINICAL_FEATURES_DIM = np.array(train_df['clinical_features'].iloc[0]).shape[0]
print(f"Διαστάσεις Κλινικών Δεδομένων: {CLINICAL_FEATURES_DIM}")

# --- 4. DATA AUGMENTATION (Επαύξηση Δεδομένων) ---
# Ενεργοποιούμε το Hair Removal, data augmentation
train_transform = transforms.Compose([
    HairRemovalTransform(), # Σχολιασμένο για ταχύτητα, ξεσχολίασε αν θες DullRazor
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ISICDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Εικόνα
        img_name = os.path.join(self.img_dir, self.df.loc[index, 'image'] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.df.loc[index, 'label'], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
            
        # Κλινικά Δεδομένα
        clinical_features = torch.tensor(self.df.loc[index, 'clinical_features'], dtype=torch.float)
        
        return image, clinical_features, label # Επιστρέφουμε τρία στοιχεία!

train_dataset = ISICDataset(train_df, IMG_DIR, train_transform)
val_dataset = ISICDataset(val_df, IMG_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. ΥΠΟΛΟΓΙΣΜΟΣ ΒΑΡΩΝ ΚΛΑΣΕΩΝ (Class Weights) ---
print("Υπολογισμός Βαρών Κλάσεων...")
y_train = train_df['label'].values
unique_classes = np.unique(y_train) 
weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
weight_dict = dict(zip(unique_classes, weights))
final_weights = [weight_dict.get(i, 0.0) for i in range(len(CLASS_NAMES))]
class_weights_tensor = torch.tensor(final_weights, dtype=torch.float).to(device)

# --- 6. ΠΟΛΥΤΡΟΠΙΚΟ ΜΟΝΤΕΛΟ (Multimodal Architecture) ---
class MultimodalNet(nn.Module):
    def __init__(self, clinical_dim, num_classes):
        super(MultimodalNet, self).__init__()
        # Branch Εικόνας: ResNet18
        self.image_branch = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Αφαιρούμε το classification layer του ResNet
        self.image_branch.fc = nn.Identity()
        
        # Branch Κλινικών Δεδομένων: MLP
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Τελικός Ταξινομητής (Fusion & Classification)
        # ResNet18 fc βγάζει 512, Clinical branch fc βγάζει 32
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image, clinical_data):
        image_features = self.image_branch(image)
        clinical_features = self.clinical_branch(clinical_data)
        
        # Concatenate (Ένωση) Features
        combined_features = torch.cat((image_features, clinical_features), dim=1)
        
        outputs = self.classifier(combined_features)
        return outputs

model = MultimodalNet(CLINICAL_FEATURES_DIM, len(CLASS_NAMES))
model = model.to(device)

# --- 7. ΕΚΠΑΙΔΕΥΣΗ ---
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nΞεκινάει η εκπαίδευση για {EPOCHS} εποχές...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, clinical_data, labels) in enumerate(train_loader):
        images = images.to(device)
        clinical_data = clinical_data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, clinical_data) # Δίνουμε και τα κλινικά δεδομένα!
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (i+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    epoch_acc = 100 * correct / total
    print(f"--> Τέλος Epoch {epoch+1} | Train Accuracy: {epoch_acc:.2f}% | Average Loss: {running_loss/len(train_loader):.4f}\n")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"🎉 Η πολυτροπική εκπαίδευση ολοκληρώθηκε! Το νέο μοντέλο σώθηκε στο: {MODEL_SAVE_PATH}")