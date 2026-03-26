import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. ΡΥΘΜΙΣΕΙΣ ---
IMG_DIR = "data/images"
LABELS_CSV = "data/ISIC_2019_Training_GroundTruth.csv"
META_CSV = "data/ISIC_2019_Training_Metadata.csv"
MODEL_PATH = "models/isic2019_resnet18_multimodal.pth"
BATCH_SIZE = 32
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# --- 2. ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ & ΑΠΟΦΥΓΗ LEAKAGE ---
print("Προετοιμασία δεδομένων και αναδημιουργία Test Set...")
labels_df = pd.read_csv(LABELS_CSV)
meta_df = pd.read_csv(META_CSV)
df = pd.merge(labels_df, meta_df, on='image')

df['target'] = df[CLASS_NAMES].idxmax(axis=1)
df['label'] = df['target'].apply(lambda x: CLASS_NAMES.index(x))
df['lesion_id'] = df['lesion_id'].fillna('unknown_lesion')

# Imputation
df['age_approx'] = df['age_approx'].fillna(df['age_approx'].mean())
df['sex'] = df['sex'].fillna('unknown')
df['anatom_site_general'] = df['anatom_site_general'].fillna('unknown')

# Διαχωρισμός (όπως ακριβώς στο train)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['lesion_id']))
train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# Fit encoders/scalers ΜΟΝΟ στο Train Set
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(train_df[['sex', 'anatom_site_general']])
scaler = StandardScaler()
scaler.fit(train_df[['age_approx']])

def create_clinical_features(dataframe):
    cat_features = encoder.transform(dataframe[['sex', 'anatom_site_general']])
    num_features = scaler.transform(dataframe[['age_approx']])
    return np.concatenate([num_features, cat_features], axis=1).tolist()

test_df['clinical_features'] = create_clinical_features(test_df)
CLINICAL_FEATURES_DIM = np.array(test_df['clinical_features'].iloc[0]).shape[0]

# --- 3. DATASET & DATALOADER ---
class ISICDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, self.df.loc[index, 'image'] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.df.loc[index, 'label'], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        clinical_features = torch.tensor(self.df.loc[index, 'clinical_features'], dtype=torch.float)
        return image, clinical_features, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ISICDataset(test_df, IMG_DIR, transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. ΠΟΛΥΤΡΟΠΙΚΟ ΜΟΝΤΕΛΟ (Multimodal Architecture) ---
class MultimodalNet(nn.Module):
    def __init__(self, clinical_dim, num_classes):
        super(MultimodalNet, self).__init__()
        self.image_branch = models.resnet18(weights=None)
        self.image_branch.fc = nn.Identity()
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image, clinical_data):
        image_features = self.image_branch(image)
        clinical_features = self.clinical_branch(clinical_data)
        combined_features = torch.cat((image_features, clinical_features), dim=1)
        outputs = self.classifier(combined_features)
        return outputs

print("Φόρτωση πολυτροπικού μοντέλου...")
model = MultimodalNet(CLINICAL_FEATURES_DIM, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# --- 5. ΑΞΙΟΛΟΓΗΣΗ ---
all_preds = []
all_labels = []

print("Ξεκινάει η μαζική αξιολόγηση...")
with torch.no_grad():
    for i, (images, clinical_data, labels) in enumerate(test_loader):
        images, clinical_data, labels = images.to(device), clinical_data.to(device), labels.to(device)
        outputs = model(images, clinical_data)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (i+1) % 20 == 0:
            print(f"Επεξεργασία Batch [{i+1}/{len(test_loader)}]")

# --- 6. ΑΠΟΤΕΛΕΣΜΑΤΑ ---
print("\n" + "="*50)
print("ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΞΙΟΛΟΓΗΣΗΣ ΠΟΛΥΤΡΟΠΙΚΟΥ ΜΟΝΤΕΛΟΥ")
print("="*50)

report = classification_report(all_labels, all_preds, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES, zero_division=0)
print(report)

# Αποθηκεύει το Report
with open("data/classification_report_multimodal.txt", "w") as f:
    f.write("ΑΠΟΤΕΛΕΣΜΑΤΑ ΠΟΛΥΤΡΟΠΙΚΟΥ (Test Set)\n")
    f.write("="*50 + "\n")
    f.write(report)

# Πίνακας Σύγχυσης (Confusion Matrix)
cm = confusion_matrix(all_labels, all_preds, labels=range(len(CLASS_NAMES)))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Πίνακας Σύγχυσης (Multimodal Model)')
plt.ylabel('Πραγματική Διάγνωση (True Label)')
plt.xlabel('Πρόβλεψη Μοντέλου (Predicted Label)')
plt.tight_layout()

# Αποθήκευση του νέου γραφήματος
cm_path = "data/confusion_matrix_multimodal.png"
plt.savefig(cm_path, dpi=300)
print(f"\n✅ Ο Πίνακας Σύγχυσης αποθηκεύτηκε στο: {cm_path}")
print("✅ Τα αναλυτικά στατιστικά αποθηκεύτηκαν στο: data/classification_report_multimodal.txt")