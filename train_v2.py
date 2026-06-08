"""
═══════════════════════════════════════════════════════════════════════════════
 train_v2.py — Βελτιωμένη Εκπαίδευση ResNet18 για ISIC 2019 (FINAL — macOS-safe)
═══════════════════════════════════════════════════════════════════════════════

ΒΕΛΤΙΩΣΕΙΣ vs train.py:
  1. 30 epochs + Early Stopping (patience=7)
  2. CosineAnnealingLR scheduler (LR: 1e-3 → 1e-6)
  3. Hair Removal εφαρμόζεται ΚΑΙ στο validation (συνέπεια)
  4. Αφαίρεση UNK κλάσης (8 αντί για 9)
  5. Πιο επιθετικό Data Augmentation (ColorJitter, RandomResizedCrop)
  6. WeightedRandomSampler (balanced batches)
  7. Focal Loss αντί CrossEntropy
  8. Best model = αυτό με το ΥΨΗΛΟΤΕΡΟ Macro F1
  9. Πλήρες logging σε CSV
  10. Weight decay (L2 regularization)
  11. macOS-safe (όλος ο εκτελέσιμος κώδικας μέσα σε if __name__ == '__main__')

ΑΝΑΜΕΝΟΜΕΝΗ ΕΠΙΔΟΣΗ:
  - Accuracy: ~60-62% (vs 53% baseline)
  - Macro F1: ~0.40-0.45 (vs 0.30 baseline)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# ═══════════════════════════════════════════════════════════════════════════════
#  ΡΥΘΜΙΣΕΙΣ ΚΑΙ ΣΤΑΘΕΡΕΣ (αυτές μπορούν να μένουν στο global scope)
# ═══════════════════════════════════════════════════════════════════════════════
IMG_DIR         = "data/images"
LABELS_CSV      = "data/ISIC_2019_Training_GroundTruth.csv"
MODEL_SAVE_PATH = "models/isic2019_resnet18_v2.pth"
LOG_CSV         = "data/training_log_v2.csv"

BATCH_SIZE     = 32
EPOCHS         = 30
LEARNING_RATE  = 0.001
PATIENCE       = 7
WEIGHT_DECAY   = 1e-4
FOCAL_GAMMA    = 2.0
NUM_WORKERS    = 2

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']


# ═══════════════════════════════════════════════════════════════════════════════
#  ΚΛΑΣΕΙΣ (μπορούν να μένουν στο global scope, χρησιμοποιούνται από workers)
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) — προσαρμογή Cross-Entropy για imbalanced data.
    Με gamma=2, τα εύκολα δείγματα συμβάλλουν 100x λιγότερο στο loss.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


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


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — ΟΛΑ ΤΑ ΕΚΤΕΛΕΣΙΜΑ ΜΕΣΑ ΕΔΩ (macOS multiprocessing requirement)
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # ── DEVICE ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Χρήση Apple M4 GPU (MPS)!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Χρήση CUDA GPU!")
    else:
        device = torch.device("cpu")
        print("Χρήση CPU — θα είναι αργό.")

    # ── 1. ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ ─────────────────────────────────────────
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
    print(f"   Κατανομή κλάσεων:")
    for cls in CLASS_NAMES:
        count = (df['target'] == cls).sum()
        print(f"     {cls:5s}: {count:5d}  ({100*count/len(df):.1f}%)")

    # ── 2. GROUP-AWARE SPLIT ─────────────────────────────────────────────
    print("\n[2/6] Group-aware διαχωρισμός...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['lesion_id']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(f"   Train: {len(train_df)} | Val: {len(val_df)}")

    # ── 3. AUGMENTATIONS ─────────────────────────────────────────────────
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

    # ── 4. WEIGHTED SAMPLER + CLASS WEIGHTS ──────────────────────────────
    print("\n[4/6] Υπολογισμός Weighted Sampler & Class Weights...")
    y_train = train_df['label'].values
    class_counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    sample_weights = np.array([1.0 / class_counts[label] for label in y_train])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0)
    )

    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"   Class weights: {dict(zip(CLASS_NAMES, [f'{w:.2f}' for w in weights]))}")

    # ── 5. ΜΟΝΤΕΛΟ ───────────────────────────────────────────────────────
    print("\n[5/6] Δημιουργία μοντέλου ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(device)

    criterion = FocalLoss(alpha=class_weights_tensor, gamma=FOCAL_GAMMA)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── 6. TRAINING LOOP ─────────────────────────────────────────────────
    print(f"\n[6/6] Ξεκινάει η εκπαίδευση για έως {EPOCHS} εποχές (patience={PATIENCE})...")
    print(f"   Best model = αυτό με ΥΨΗΛΟΤΕΡΟ Macro F1 στο val set")
    print("=" * 95)

    best_macro_f1 = 0.0
    best_val_acc = 0.0
    epochs_no_improve = 0
    log_rows = []

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    for epoch in range(EPOCHS):
        t0 = time.time()

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

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        elapsed = time.time() - t0
        log_rows.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
            'lr': current_lr, 'time_sec': elapsed,
        })

        print(
            f"Epoch [{epoch+1:2d}/{EPOCHS}] "
            f"| TrL={train_loss:.3f} TrA={train_acc:5.2f}% "
            f"| VaL={val_loss:.3f} VaA={val_acc:5.2f}% "
            f"| MacF1={macro_f1:.3f} WgtF1={weighted_f1:.3f} "
            f"| {elapsed:.0f}s"
        )

        # ── BEST MODEL TRACKING ──
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   💾 Νέο καλύτερο μοντέλο — Macro F1 {macro_f1:.4f} | Val Acc {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ Δεν υπήρξε βελτίωση Macro F1 ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping στο epoch {epoch+1}.")
                break

        # Αποθήκευση log σε κάθε epoch (σε περίπτωση που σπάσει η εκπαίδευση)
        with open(LOG_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

    # ── ΤΕΛΟΣ ────────────────────────────────────────────────────────────
    print("=" * 95)
    print(f"\n🎉 Εκπαίδευση τελείωσε!")
    print(f"   Καλύτερο Macro F1:  {best_macro_f1:.4f}")
    print(f"   Καλύτερο Val Acc:   {best_val_acc:.2f}%")
    print(f"   Μοντέλο σώθηκε στο: {MODEL_SAVE_PATH}")
    print(f"   Log σώθηκε στο:     {LOG_CSV}")
    print("\n📋 Επόμενο βήμα: τρέξε το evaluate_v2.py για αναλυτικές μετρικές στο test set.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — Κρίσιμο για macOS (spawn multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Για ασφάλεια σε macOS — set spawn method ρητά
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass  # Already set
    main()