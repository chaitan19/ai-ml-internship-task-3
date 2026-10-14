"""Generate charts from actual held-out test predictions."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

PALETTE = {"No": "#4063D8", "Yes": "#E66B4E"}
def create_model_charts(results, y_test, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    y_binary = (y_test == "Yes").astype(int)
    for result in results:
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
        ax.set(title=f"{result['model']} - confusion matrix", xlabel="Predicted attrition", ylabel="Actual attrition")
        fig.tight_layout(); fig.savefig(output_dir / f"confusion_matrix_{slug(result['model'])}.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.3, 5.6))
    for result in results:
        fpr, tpr, _ = roc_curve(y_binary, result["y_score"])
        ax.plot(fpr, tpr, lw=2, label=f"{result['model']} (AUC {result['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set(xlim=(0,1), ylim=(0,1.02), xlabel="False positive rate", ylabel="True positive rate", title="ROC curves - held-out test set")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=170); plt.close(fig)
    # Count matrix and ROC plot use identical candidate order.
def slug(text):
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")
