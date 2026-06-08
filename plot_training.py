"""
═══════════════════════════════════════════════════════════════════════════════
 plot_training.py — Γραφήματα Εκπαίδευσης για την Πτυχιακή
═══════════════════════════════════════════════════════════════════════════════

Διαβάζει τα CSV logs (training_log_v2.csv, training_log_v3.csv) και παράγει
γραφήματα υψηλής ποιότητας (300 dpi) έτοιμα για ένθεση στην αναφορά:

  1. training_curves_v3.png    — Loss & Accuracy ανά epoch (v3)
  2. metrics_curves_v3.png     — Macro F1 / Weighted F1 / Balanced Acc ανά epoch
  3. comparison_v2_v3.png      — Σύγκριση v2 vs v3 (αν υπάρχουν και τα δύο logs)

ΧΡΗΣΗ:
  python plot_training.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Στυλ για επαγγελματική εμφάνιση
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 100,
})

LOG_V2 = "data/training_log_v2.csv"
LOG_V3 = "data/training_log_v3.csv"
OUT_DIR = "data"

# Χρωματική παλέτα
C_TRAIN = "#2563EB"   # μπλε
C_VAL   = "#DC2626"   # κόκκινο
C_F1    = "#059669"   # πράσινο
C_WF1   = "#D97706"   # πορτοκαλί
C_BAL   = "#7C3AED"   # μωβ


def plot_v3_curves():
    if not os.path.exists(LOG_V3):
        print(f"Δεν βρέθηκε {LOG_V3} — παράλειψη γραφημάτων v3.")
        return
    df = pd.read_csv(LOG_V3)

    # ── ΓΡΑΦΗΜΑ 1: Loss & Accuracy ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['epoch'], df['train_loss'], 'o-', color=C_TRAIN, label='Train Loss', markersize=4)
    ax1.plot(df['epoch'], df['val_loss'], 's-', color=C_VAL, label='Val Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Καμπύλη Σφάλματος (Loss) ανά Epoch', fontweight='bold')
    ax1.legend()

    ax2.plot(df['epoch'], df['train_acc'], 'o-', color=C_TRAIN, label='Train Accuracy', markersize=4)
    ax2.plot(df['epoch'], df['val_acc'], 's-', color=C_VAL, label='Val Accuracy', markersize=4)
    if 'balanced_acc' in df.columns:
        ax2.plot(df['epoch'], df['balanced_acc'], '^-', color=C_BAL, label='Balanced Accuracy', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Καμπύλη Ακρίβειας (Accuracy) ανά Epoch', fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/training_curves_v3.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{OUT_DIR}/training_curves_v3.png")

    # ── ΓΡΑΦΗΜΑ 2: F1 Metrics ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['macro_f1'], 'o-', color=C_F1, label='Macro F1', markersize=5, linewidth=2)
    ax.plot(df['epoch'], df['weighted_f1'], 's-', color=C_WF1, label='Weighted F1', markersize=5, linewidth=2)
    # Σημάδεψε το καλύτερο macro F1
    best_idx = df['macro_f1'].idxmax()
    ax.axvline(df.loc[best_idx, 'epoch'], color='gray', linestyle='--', alpha=0.5)
    ax.annotate(f"Best Macro F1\n{df.loc[best_idx,'macro_f1']:.3f} @ ep{int(df.loc[best_idx,'epoch'])}",
                xy=(df.loc[best_idx, 'epoch'], df.loc[best_idx, 'macro_f1']),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.4', fc='#FEF3C7', ec='gray'),
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Εξέλιξη F1-Score ανά Epoch (v3)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/metrics_curves_v3.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{OUT_DIR}/metrics_curves_v3.png")


def plot_comparison():
    """Σύγκριση v2 vs v3 αν υπάρχουν και τα δύο."""
    if not (os.path.exists(LOG_V2) and os.path.exists(LOG_V3)):
        print("Χρειάζονται και τα δύο logs (v2, v3) για το comparison — παράλειψη.")
        return
    d2 = pd.read_csv(LOG_V2)
    d3 = pd.read_csv(LOG_V3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(d2['epoch'], d2['val_acc'], 's-', color="#9CA3AF", label='v2 Val Acc', markersize=4)
    ax1.plot(d3['epoch'], d3['val_acc'], 'o-', color=C_VAL, label='v3 Val Acc', markersize=4)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Σύγκριση Val Accuracy: v2 vs v3', fontweight='bold')
    ax1.legend()

    ax2.plot(d2['epoch'], d2['macro_f1'], 's-', color="#9CA3AF", label='v2 Macro F1', markersize=4)
    ax2.plot(d3['epoch'], d3['macro_f1'], 'o-', color=C_F1, label='v3 Macro F1', markersize=4)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Macro F1')
    ax2.set_title('Σύγκριση Macro F1: v2 vs v3', fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/comparison_v2_v3.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{OUT_DIR}/comparison_v2_v3.png")


if __name__ == '__main__':
    print("Δημιουργία γραφημάτων...")
    plot_v3_curves()
    plot_comparison()
    print("\n🎉 Έτοιμα! Τα γραφήματα είναι στον φάκελο data/ — έτοιμα για την αναφορά.")
