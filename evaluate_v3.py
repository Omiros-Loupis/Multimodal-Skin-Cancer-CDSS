"""
═══════════════════════════════════════════════════════════════════════════════
 evaluate_v3.py — Πλήρης Αξιολόγηση του v3 μοντέλου στο Test Set
═══════════════════════════════════════════════════════════════════════════════

Παράγει:
  1. classification_report_v3.txt  (precision/recall/f1 ανά κλάση)
  2. confusion_matrix_v3.png        (πίνακας σύγχυσης)
  3. confusion_matrix_v3_norm.png   (κανονικοποιημένος — % ανά γραμμή)
  4. metrics_summary_v3.txt         (accuracy, macro/weighted/balanced)

ΧΡΗΣΗ:
  python evaluate_v3.py            # κανονική αξιολόγηση
  python evaluate_v3.py --tta      # με Test-Time Augmentation (+1-2% συνήθως)

ΣΗΜΑΝΤΙΚΟ: Χρησιμοποιεί ΤΟΝ ΙΔΙΟ διαχωρισμό (random_state=42) με το train_v3.py,
ώστε το "test set" εδώ να είναι το ίδιο val set που ΔΕΝ είδε η εκπαίδευση.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ── ΡΥΘΜΙΣΕΙΣ ──────────────────────────────────────────────────────────────────
IMG_DIR     = "data/images"
LABELS_CSV  = "data/ISIC_2019_Training_GroundTruth.csv"
MODEL_PATH  = "models/isic2019_resnet18_v3.pth"
BATCH_SIZE  = 32
NUM_WORKERS = 2
USE_TTA     = '--tta' in sys.argv

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_FULL = {
    'MEL': 'Μελάνωμα', 'NV': 'Σπίλος', 'BCC': 'Βασικοκυτταρικό',
    'AK': 'Ακτινική Κεράτωση', 'BKL': 'Καλοήθης Κεράτωση', 'DF': 'Δερματοΐνωμα',
    'VASC': 'Αγγειακή Βλάβη', 'SCC': 'Ακανθοκυτταρικό',
}


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


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | TTA: {'ON' if USE_TTA else 'OFF'}")

    # ── ΔΕΔΟΜΕΝΑ (ίδιος split με train_v3) ──
    print("\nΠροετοιμασία test set (ίδιος διαχωρισμός με την εκπαίδευση)...")
    df = pd.read_csv(LABELS_CSV)
    cols_to_check = CLASS_NAMES + ['UNK'] if 'UNK' in df.columns else CLASS_NAMES
    df['target'] = df[cols_to_check].idxmax(axis=1)
    df = df[df['target'].isin(CLASS_NAMES)].reset_index(drop=True)
    df['label'] = df['target'].apply(lambda x: CLASS_NAMES.index(x))
    if 'lesion_id' in df.columns:
        df['lesion_id'] = df['lesion_id'].fillna('unknown_lesion')
    else:
        df['lesion_id'] = df['image']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(df, groups=df['lesion_id']))
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"   Test set: {len(test_df)} εικόνες")

    val_transform = transforms.Compose([
        HairRemovalTransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # TTA transforms: εφαρμόζονται μόνο αν USE_TTA
    tta_transforms = [
        val_transform,
        transforms.Compose([
            HairRemovalTransform(), transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            HairRemovalTransform(), transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    ]

    # ── ΜΟΝΤΕΛΟ ──
    print("Φόρτωση μοντέλου v3...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    if not os.path.exists(MODEL_PATH):
        print(f"Δεν βρέθηκε: {MODEL_PATH}")
        sys.exit(1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # ── ΑΞΙΟΛΟΓΗΣΗ ──
    print("Αξιολόγηση...")
    all_preds, all_labels = [], []

    if not USE_TTA:
        test_loader = DataLoader(
            ISICDataset(test_df, IMG_DIR, val_transform),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                if (i + 1) % 20 == 0:
                    print(f"   Batch [{i+1}/{len(test_loader)}]")
    else:
        # TTA: average softmax πάνω από 3 transforms
        loaders = [
            DataLoader(ISICDataset(test_df, IMG_DIR, t), batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=NUM_WORKERS)
            for t in tta_transforms
        ]
        n = len(test_df)
        prob_sum = np.zeros((n, len(CLASS_NAMES)))
        labels_arr = None
        for ti, loader in enumerate(loaders):
            print(f"   TTA pass {ti+1}/{len(loaders)}...")
            offset = 0
            lab_tmp = []
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    probs = torch.softmax(model(images), dim=1).cpu().numpy()
                    prob_sum[offset:offset+len(probs)] += probs
                    offset += len(probs)
                    lab_tmp.extend(labels.numpy())
            labels_arr = np.array(lab_tmp)
        all_preds = prob_sum.argmax(axis=1).tolist()
        all_labels = labels_arr.tolist()

    # ── ΜΕΤΡΙΚΕΣ ──
    acc = 100 * accuracy_score(all_labels, all_preds)
    bal_acc = 100 * balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    suffix = "_tta" if USE_TTA else ""
    print("\n" + "=" * 60)
    print(f"ΑΠΟΤΕΛΕΣΜΑΤΑ v3 {'(με TTA)' if USE_TTA else ''}")
    print("=" * 60)
    print(f"Accuracy:          {acc:.2f}%")
    print(f"Balanced Accuracy: {bal_acc:.2f}%")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")
    print("=" * 60)

    report = classification_report(
        all_labels, all_preds, labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, zero_division=0
    )
    print(report)

    with open(f"data/metrics_summary_v3{suffix}.txt", "w") as f:
        f.write(f"ΑΠΟΤΕΛΕΣΜΑΤΑ v3 {'(TTA)' if USE_TTA else ''}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:          {acc:.2f}%\n")
        f.write(f"Balanced Accuracy: {bal_acc:.2f}%\n")
        f.write(f"Macro F1:          {macro_f1:.4f}\n")
        f.write(f"Weighted F1:       {weighted_f1:.4f}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    with open(f"data/classification_report_v3{suffix}.txt", "w") as f:
        f.write(f"ΑΠΟΤΕΛΕΣΜΑΤΑ v3 (Test Set){' με TTA' if USE_TTA else ''}\n")
        f.write("=" * 50 + "\n")
        f.write(report)

    # ── CONFUSION MATRIX (raw) ──
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(CLASS_NAMES)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Πλήθος'})
    plt.title(f'Πίνακας Σύγχυσης v3{" (TTA)" if USE_TTA else ""}', fontsize=14, pad=12)
    plt.ylabel('Πραγματική Διάγνωση', fontsize=12)
    plt.xlabel('Πρόβλεψη Μοντέλου', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"data/confusion_matrix_v3{suffix}.png", dpi=200)
    plt.close()

    # ── CONFUSION MATRIX (normalized ανά γραμμή = recall) ──
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Ποσοστό (recall ανά κλάση)'}, vmin=0, vmax=1)
    plt.title(f'Κανονικοποιημένος Πίνακας Σύγχυσης v3{" (TTA)" if USE_TTA else ""}',
              fontsize=14, pad=12)
    plt.ylabel('Πραγματική Διάγνωση', fontsize=12)
    plt.xlabel('Πρόβλεψη Μοντέλου', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"data/confusion_matrix_v3_norm{suffix}.png", dpi=200)
    plt.close()

    print(f"\n Αρχεία αποθηκεύτηκαν στο data/ (suffix='{suffix}'):")
    print(f"   • classification_report_v3{suffix}.txt")
    print(f"   • metrics_summary_v3{suffix}.txt")
    print(f"   • confusion_matrix_v3{suffix}.png")
    print(f"   • confusion_matrix_v3_norm{suffix}.png")


if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass
    main()
