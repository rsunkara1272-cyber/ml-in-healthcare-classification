"""
Classification – Breast Cancer Wisconsin
=========================================
Dataset : sklearn built-in (no download needed)
Model   : Random Forest + Logistic Regression baseline
Plots   : Confusion Matrix, ROC Curve, Feature Importance, Precision-Recall
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import logging, os

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT_DIR = "outputs/classification"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

STYLE = {
    "bg":      "#0A0E1A",
    "panel":   "#141928",
    "accent1": "#A78BFA",
    "accent2": "#F472B6",
    "accent3": "#34D399",
    "accent4": "#FBBF24",
    "text":    "#F1F5F9",
    "subtext": "#94A3B8",
}

def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  STYLE["bg"],
        "axes.facecolor":    STYLE["panel"],
        "axes.edgecolor":    STYLE["subtext"],
        "axes.labelcolor":   STYLE["text"],
        "xtick.color":       STYLE["subtext"],
        "ytick.color":       STYLE["subtext"],
        "text.color":        STYLE["text"],
        "grid.color":        "#1E2A45",
        "grid.alpha":        0.6,
        "font.family":       "DejaVu Sans",
    })

# ── Data ──────────────────────────────────────

def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    log.info("Breast Cancer dataset: %d samples, %d features, classes=%s",
             len(X), X.shape[1], list(data.target_names))
    return X, y, data.target_names, data.feature_names

# ── Models ────────────────────────────────────

def train_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_s, y_train)

    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                 min_samples_split=2, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    results = {}
    for name, model, Xtr, Xte in [
        ("Logistic Regression", lr, X_train_s, X_test_s),
        ("Random Forest",       rf, X_train,   X_test),
    ]:
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        acc = (y_pred == y_test).mean()
        cm  = confusion_matrix(y_test, y_pred)

        results[name] = dict(model=model, y_pred=y_pred, y_proba=y_proba,
                              fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                              prec=prec, rec=rec, ap=ap, acc=acc, cm=cm)
        log.info("%-22s  Acc=%.4f  AUC=%.4f  AP=%.4f", name, acc, roc_auc, ap)

    return results, scaler

# ── Plots ─────────────────────────────────────

def plot_all(results, feature_names, class_names):
    set_dark_style()
    rf = results["Random Forest"]
    lr = results["Logistic Regression"]

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(STYLE["bg"])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Confusion Matrix (RF) ──
    ax1 = fig.add_subplot(gs[0, 0])
    cm = rf["cm"]
    im = ax1.imshow(cm, cmap="Blues", aspect="auto")
    ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
    ax1.set_xticklabels(class_names, fontsize=10)
    ax1.set_yticklabels(class_names, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i,j]), ha="center", va="center",
                     fontsize=16, fontweight="bold",
                     color="white" if cm[i,j] > cm.max()/2 else STYLE["text"])
    ax1.set(title=f"Confusion Matrix – Random Forest\nAccuracy: {rf['acc']:.4f}",
            xlabel="Predicted", ylabel="Actual")

    # ── 2. ROC Curve ──
    ax2 = fig.add_subplot(gs[0, 1])
    for name, r, color in [
        ("Random Forest",       rf, STYLE["accent1"]),
        ("Logistic Regression", lr, STYLE["accent2"]),
    ]:
        ax2.plot(r["fpr"], r["tpr"], color=color, lw=2,
                 label=f"{name}  AUC={r['roc_auc']:.4f}")
    ax2.plot([0,1],[0,1],"--", color=STYLE["subtext"], lw=1, label="Random")
    ax2.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # ── 3. Feature Importance (top 10) ──
    ax3 = fig.add_subplot(gs[1, 0])
    imp = rf["model"].feature_importances_
    top_idx = np.argsort(imp)[-10:]
    colors = [STYLE["accent3"] if i == top_idx[-1] else STYLE["accent1"] for i in top_idx]
    ax3.barh(np.array(feature_names)[top_idx], imp[top_idx], color=colors, edgecolor="none")
    ax3.set(title="Top 10 Feature Importances (RF)", xlabel="Importance")
    ax3.grid(True, alpha=0.3, axis="x")

    # ── 4. Precision-Recall Curve ──
    ax4 = fig.add_subplot(gs[1, 1])
    for name, r, color in [
        ("Random Forest",       rf, STYLE["accent1"]),
        ("Logistic Regression", lr, STYLE["accent2"]),
    ]:
        ax4.plot(r["rec"], r["prec"], color=color, lw=2,
                 label=f"{name}  AP={r['ap']:.4f}")
    ax4.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    fig.text(0.5, 0.97,
             f"Breast Cancer Classification  |  Random Forest  AUC={rf['roc_auc']:.4f}  Acc={rf['acc']:.4f}",
             ha="center", va="top", fontsize=13, fontweight="bold", color=STYLE["text"])

    path = f"{OUTPUT_DIR}/classification_linkedin.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    log.info("Saved → %s", path)

# ── Main ──────────────────────────────────────

def main():
    log.info("═"*55)
    log.info("  Classification – Breast Cancer Wisconsin")
    log.info("═"*55)

    X, y, class_names, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    results, scaler = train_models(X_train, X_test, y_train, y_test)
    plot_all(results, feature_names, class_names)

    rf = results["Random Forest"]
    print(f"\n{'═'*45}")
    print(f"  RANDOM FOREST RESULTS")
    print(f"{'═'*45}")
    print(f"  Accuracy : {rf['acc']:.4f}")
    print(f"  ROC AUC  : {rf['roc_auc']:.4f}")
    print(f"  Avg Prec : {rf['ap']:.4f}")
    print(f"\n{classification_report(y_test, rf['y_pred'], target_names=class_names)}")
    log.info("Done ✓")

if __name__ == "__main__":
    main()
