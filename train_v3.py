"""
═══════════════════════════════════════════════════════════════════════════════
 train_v3.py — Ισορροπημένη Εκπαίδευση ResNet18 για ISIC 2019
═══════════════════════════════════════════════════════════════════════════════

ΔΙΟΡΘΩΣΗ vs train_v2.py:
  Το v2 έκανε ΤΡΙΠΛΟ balancing (Sampler + class_weights + Focal) → το accuracy
  κατέρρευσε στο 42% επειδή το μοντέλο "τιμωρήθηκε" υπερβολικά για την NV.

ΑΛΛΑΓΕΣ v3:
  1. ΜΟΝΟ ΕΝΑΣ μηχανισμός balancing: "soft" WeightedRandomSampler (√ των βαρών)
     → δίνει βάρος στις σπάνιες κλάσεις ΧΩΡΙΣ να εξαφανίζει τις πλειοψηφικές
  2. ΚΑΘΑΡΗ CrossEntropy + Label Smoothing 0.1 (όχι Focal, όχι class_weights)
     → Label Smoothing μειώνει το overfitting (gap train/val)
  3. FREEZE του ResNet τα πρώτα 3 epochs (μόνο το fc εκπαιδεύεται)
     → σταθεροποιεί το νέο classification layer πριν "ξεκλειδώσει" το backbone
  4. Αποθήκευση ΚΑΙ best (Macro F1) ΚΑΙ last model
  5. Μετράμε balanced accuracy επιπλέον (πιο δίκαιη μετρική)

ΣΤΟΧΟΣ: Accuracy ~58-62% ΚΑΙ Macro F1 ~0.40+ ταυτόχρονα (ισορροπία)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score

# ── ΡΥΘΜΙΣΕΙΣ ──────────────────────────────────────────────────────────────────
IMG_DIR         = "data/images"
LABELS_CSV      = "data/ISIC_2019_Training_GroundTruth.csv"
MODEL_SAVE_PATH = "models/isic2019_resnet18_v3.pth"
MODEL_LAST_PATH = "models/isic2019_resnet18_v3_last.pth"
LOG_CSV         = "data/training_log_v3.csv"

BATCH_SIZE      = 32
EPOCHS          = 30
LEARNING_RATE   = 0.001
PATIENCE        = 8
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
FREEZE_EPOCHS   = 3          # Πόσα epochs να μείνει "παγωμένο" το ResNet backbone
NUM_WORKERS     = 2
SAMPLER_POWER   = 0.5        # 0.5 = soft (√), 1.0 = πλήρες balancing (όπως v2)

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']


# ── HAIR REMOVAL ───────────────────────────────────────────────────────────────
class HairRemovalTransform(object):
    def __call__(self, img):
        cv_img = np.array(img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        img_inpainted = cv2.inpaint(cv_img, mask, 1, cv2.INPAINT_TELEA)
        img_inpainted = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_inpainted)


# ── DATASET ────────────────────────────────────────────────────────────────────
class ISICDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.loc[idx, 'image'] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.df.loc[idx, 'label'], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


# ── FREEZE / UNFREEZE HELPERS ──────────────────────────────────────────────────
def set_backbone_frozen(model, frozen: bool):
    """Παγώνει/ξεπαγώνει όλα τα layers ΕΚΤΟΣ από το fc."""
    for name, param in model.named_parameters():
        if not name.startswith('fc.'):
            param.requires_grad = not frozen


def main():
    # ── DEVICE ──
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Χρήση Apple M4 GPU (MPS)!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Χρήση CUDA GPU!")
    else:
        device = torch.device("cpu")
        print("⚠️  Χρήση CPU.")

    # ── 1. ΔΕΔΟΜΕΝΑ ──
    print("\n[1/6] Φόρτωση δεδομένων...")
    df = pd.read_csv(LABELS_CSV)
    cols_to_check = CLASS_NAMES + ['UNK'] if 'UNK' in df.columns else CLASS_NAMES
    df['target'] = df[cols_to_check].idxmax(axis=1)
    df = df[df['target'].isin(CLASS_NAMES)].reset_index(drop=True)
    df['label'] = df['target'].apply(lambda x: CLASS_NAMES.index(x))
    if 'lesion_id' in df.columns:
        df['lesion_id'] = df['lesion_id'].fillna('unknown_lesion')
    else:
        df['lesion_id'] = df['image']

    print(f"   Συνολικά δείγματα: {len(df)}")
    for cls in CLASS_NAMES:
        count = (df['target'] == cls).sum()
        print(f"     {cls:5s}: {count:5d}  ({100*count/len(df):.1f}%)")

    # ── 2. SPLIT ──
    print("\n[2/6] Group-aware διαχωρισμός...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['lesion_id']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(f"   Train: {len(train_df)} | Val: {len(val_df)}")

    # ── 3. AUGMENTATIONS ──
    print("\n[3/6] Ορισμός Augmentations...")
    train_transform = transforms.Compose([
        HairRemovalTransform(),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        HairRemovalTransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = ISICDataset(train_df, IMG_DIR, train_transform)
    val_dataset = ISICDataset(val_df, IMG_DIR, val_transform)

    # ── 4. SOFT WEIGHTED SAMPLER (ΜΟΝΟΣ μηχανισμός balancing) ──
    print("\n[4/6] Υπολογισμός Soft Weighted Sampler...")
    y_train = train_df['label'].values
    class_counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    # SOFT balancing: βάρος = 1 / (count ^ power), με power=0.5 (√)
    # Αυτό δίνει βάρος στις σπάνιες ΧΩΡΙΣ να εξαφανίζει τις πλειοψηφικές
    class_sample_weight = 1.0 / (class_counts ** SAMPLER_POWER)
    sample_weights = np.array([class_sample_weight[label] for label in y_train])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True,
    )

    # Δείξε την effective κατανομή που θα "βλέπει" το μοντέλο
    eff = class_sample_weight * class_counts
    eff = eff / eff.sum()
    print("   Effective κατανομή ανά batch (soft):")
    for cls, e in zip(CLASS_NAMES, eff):
        print(f"     {cls:5s}: {100*e:.1f}%")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0)
    )

    # ── 5. ΜΟΝΤΕΛΟ ──
    print("\n[5/6] Δημιουργία μοντέλου ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(device)

    # ΚΑΘΑΡΗ CrossEntropy + Label Smoothing (ΧΩΡΙΣ class_weights — ο sampler αρκεί)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── 6. TRAINING ──
    print(f"\n[6/6] Εκπαίδευση έως {EPOCHS} epochs (patience={PATIENCE}, freeze πρώτα {FREEZE_EPOCHS})...")
    print("=" * 100)

    best_macro_f1 = 0.0
    best_val_acc = 0.0
    epochs_no_improve = 0
    log_rows = []
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Ξεκινάμε με ΠΑΓΩΜΕΝΟ backbone
    backbone_frozen = True
    set_backbone_frozen(model, frozen=True)
    print(f"   🔒 Backbone ΠΑΓΩΜΕΝΟ — εκπαιδεύεται μόνο το fc layer για {FREEZE_EPOCHS} epochs")

    for epoch in range(EPOCHS):
        t0 = time.time()

        # Ξεπάγωμα μετά τα FREEZE_EPOCHS
        if backbone_frozen and epoch >= FREEZE_EPOCHS:
            set_backbone_frozen(model, frozen=False)
            backbone_frozen = False
            print(f"   🔓 Epoch {epoch+1}: Backbone ΞΕΠΑΓΩΘΗΚΕ — εκπαιδεύεται ολόκληρο το δίκτυο")

        # ── TRAIN ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # ── VALIDATE ──
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)
        val_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        bal_acc = 100 * balanced_accuracy_score(all_labels, all_preds)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        elapsed = time.time() - t0

        log_rows.append({
            'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'balanced_acc': bal_acc,
            'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
            'lr': current_lr, 'time_sec': elapsed,
        })

        print(
            f"Epoch [{epoch+1:2d}/{EPOCHS}] "
            f"| TrA={train_acc:5.2f}% | VaA={val_acc:5.2f}% BalA={bal_acc:5.2f}% "
            f"| MacF1={macro_f1:.3f} WgtF1={weighted_f1:.3f} | {elapsed:.0f}s"
        )

        # Αποθήκευση last
        torch.save(model.state_dict(), MODEL_LAST_PATH)

        # Best βάσει Macro F1
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   💾 Best — MacF1 {macro_f1:.4f} | VaA {val_acc:.2f}% | BalA {bal_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ Χωρίς βελτίωση ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"\n Early stopping στο epoch {epoch+1}.")
                break

        with open(LOG_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

    print("=" * 100)
    print(f"\n🎉 Τέλος! Best Macro F1: {best_macro_f1:.4f} | Best Val Acc: {best_val_acc:.2f}%")
    print(f"   Best model: {MODEL_SAVE_PATH}")
    print(f"   Last model: {MODEL_LAST_PATH}")
    print(f"   Log: {LOG_CSV}")
    print("\n📋 Επόμενο: evaluate_v3.py στο test set.")


if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass
    main()
