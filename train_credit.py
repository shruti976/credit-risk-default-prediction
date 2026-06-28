"""
Credit-card default prediction with interpretability.

Dataset: UCI "Default of Credit Card Clients" (30,000 accounts, ~22% default).
Pipeline: clean undocumented category codes -> stratified split -> train a
logistic-regression baseline and a gradient-boosted (XGBoost) model with class
imbalance handling -> evaluate with imbalance-aware metrics (ROC-AUC, PR-AUC),
probability calibration, and SHAP explanations (global drivers + per-applicant).
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix, brier_score_loss,
                             f1_score, recall_score, precision_score)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import shap

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets"); os.makedirs(ASSETS, exist_ok=True)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
TARGET = "default payment next month"


def load():
    df = pd.read_excel(os.path.join(HERE, "data", "credit.xls"), header=1).drop(columns=["ID"])
    # collapse undocumented category codes into the "other" bucket
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    df = df.rename(columns={"PAY_0": "PAY_1"})
    y = df[TARGET].astype(int); X = df.drop(columns=[TARGET])
    return X, y


def evaluate(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"ROC_AUC": round(roc_auc_score(y, p), 4),
            "PR_AUC": round(average_precision_score(y, p), 4),
            "Brier": round(brier_score_loss(y, p), 4),
            "F1@0.5": round(f1_score(y, pred), 4),
            "Recall@0.5": round(recall_score(y, pred), 4),
            "Precision@0.5": round(precision_score(y, pred), 4)}


def main():
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"train {Xtr.shape}  test {Xte.shape}  default rate={y.mean():.3f}")

    # ---- logistic regression baseline (scaled, class-balanced) ----
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    lr.fit(sc.transform(Xtr), ytr)
    p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]

    # ---- gradient boosting (XGBoost) with imbalance weighting ----
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
        eval_metric="aucpr", random_state=SEED, n_jobs=4)
    model.fit(Xtr, ytr)
    p_xgb = model.predict_proba(Xte)[:, 1]

    results = {"LogisticRegression": evaluate(yte, p_lr),
               "XGBoost": evaluate(yte, p_xgb)}
    for k, v in results.items():
        print(f"{k:20s} ROC-AUC={v['ROC_AUC']}  PR-AUC={v['PR_AUC']}  Recall={v['Recall@0.5']}")
    json.dump(results, open(os.path.join(RESULTS, "metrics.json"), "w"), indent=2)
    pd.DataFrame(results).T.to_csv(os.path.join(RESULTS, "metrics.csv"))

    make_curves(yte, p_lr, p_xgb)
    make_confusion(yte, p_xgb)
    make_calibration(yte, p_lr, p_xgb)
    make_shap(model, Xte, yte, p_xgb)
    print("\nSaved metrics + figures.\n" + json.dumps(results, indent=2))


def make_curves(y, p_lr, p_xgb):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    for name, p, c in [("LogReg", p_lr, "#9aa7b8"), ("XGBoost", p_xgb, "#56a98c")]:
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y,p):.3f})", color=c, lw=2)
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, label=f"{name} (AP {average_precision_score(y,p):.3f})", color=c, lw=2)
    a1.plot([0, 1], [0, 1], "k--", alpha=0.4); a1.set_title("ROC curve")
    a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate"); a1.legend(); a1.grid(alpha=0.3)
    a2.axhline(y.mean(), ls="--", color="k", alpha=0.4, label=f"base rate {y.mean():.2f}")
    a2.set_title("Precision-Recall curve"); a2.set_xlabel("recall"); a2.set_ylabel("precision")
    a2.legend(); a2.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "roc_pr_curves.png")); plt.close(fig)


def make_confusion(y, p):
    cm = confusion_matrix(y, (p >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    im = ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["pred: no default", "pred: default"])
    ax.set_yticks([0, 1], ["actual: no default", "actual: default"])
    ax.set_title("XGBoost confusion matrix (threshold 0.5)"); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "confusion_matrix.png")); plt.close(fig)


def make_calibration(y, p_lr, p_xgb):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for name, p, c in [("LogReg", p_lr, "#9aa7b8"), ("XGBoost", p_xgb, "#56a98c")]:
        frac, mean_pred = calibration_curve(y, p, n_bins=10)
        ax.plot(mean_pred, frac, "o-", label=f"{name} (Brier {brier_score_loss(y,p):.3f})", color=c)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfectly calibrated")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed default frequency")
    ax.set_title("Probability calibration"); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "calibration.png")); plt.close(fig)


def make_shap(model, Xte, yte, p_xgb):
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xte)
    # global: beeswarm
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("SHAP: drivers of predicted default risk"); plt.tight_layout()
    plt.gcf().savefig(os.path.join(ASSETS, "shap_summary.png"), bbox_inches="tight"); plt.close()
    # global: mean |SHAP| bar
    plt.figure()
    shap.plots.bar(sv, max_display=12, show=False)
    plt.title("Top default-risk features (mean |SHAP|)"); plt.tight_layout()
    plt.gcf().savefig(os.path.join(ASSETS, "shap_importance.png"), bbox_inches="tight"); plt.close()
    # local: a high-risk applicant the model flags
    idx = int(np.argmax(p_xgb))
    plt.figure()
    shap.plots.waterfall(sv[idx], max_display=12, show=False)
    plt.title("Why this applicant was flagged high-risk"); plt.tight_layout()
    plt.gcf().savefig(os.path.join(ASSETS, "shap_local.png"), bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
