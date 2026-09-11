from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

def save_confusion_matrix(y_true, y_pred, output: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Benign / control","Cancer-positive"], yticklabels=["Benign / control","Cancer-positive"], xlabel="Predicted class", ylabel="True class", title="ben alpha 1.0 — confusion matrix\nSynthetic demonstration")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)

def save_roc_curve(y_true, probability, output: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probability)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot(fpr, tpr, linewidth=2, label="ben alpha 1.0")
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1, label="Chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ben alpha 1.0 — ROC curve\nSynthetic demonstration", xlim=(0,1), ylim=(0,1))
    ax.grid(alpha=0.25); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)
