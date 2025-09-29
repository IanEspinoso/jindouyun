import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, classification_report, roc_curve, precision_recall_curve
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


HAS_LGBM = True
try:
    from lightgbm import LGBMClassifier
except Exception:
    HAS_LGBM = False

# Paths & data loading
csv_path = "./data/cloudwalk_transactional-sample.csv"
output_dir = Path("./data/")
output_dir.mkdir(parents=True, exist_ok=True)

def load_dataframe(path):
    df = pd.read_csv(path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    # Ensure target is 0/1
    if df["has_cbk"].dtype != int:
        df["has_cbk"] = df["has_cbk"].astype(int)
    # basic safety
    df = df.dropna(subset=["transaction_date", "transaction_amount"]).reset_index(drop=True)
    return df

# Simple feature builder
class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Very small, readable feature set for a quick bake-off."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["hour"] = df["transaction_date"].dt.hour

        # Simple rules from your project notes
        df["amount_gt_1000"]    = (df["transaction_amount"] > 1000).astype(int)
        df["is_business_hour"]  = df["hour"].between(9, 17).astype(int)

        df["tx_per_card"]   = df.groupby("card_number")["transaction_id"].transform("count")
        df["tx_per_user"]   = df.groupby("user_id")["transaction_id"].transform("count")
        df["tx_per_device"] = df.groupby("device_id")["transaction_id"].transform("count")

        df["card_tx_gt1"]   = (df["tx_per_card"] > 1).astype(int)
        df["user_tx_gt1"]   = (df["tx_per_user"] > 1).astype(int)
        df["device_tx_gt1"] = (df["tx_per_device"] > 1).astype(int)

        df["merchants_per_device"] = df.groupby("device_id")["merchant_id"].transform("nunique")
        df["device_merch_gt1"]     = (df["merchants_per_device"] > 1).astype(int)

        df["merchants_per_user"] = df.groupby("user_id")["merchant_id"].transform("nunique")
        df["user_merch_gt1"]     = (df["merchants_per_user"] > 1).astype(int)

        df["merchants_per_card"] = df.groupby("card_number")["merchant_id"].transform("nunique")
        df["card_merch_gt1"]     = (df["merchants_per_card"] > 1).astype(int)

        df["users_per_card"]     = df.groupby("card_number")["user_id"].transform("nunique")
        df["card_users_gt1"]     = (df["users_per_card"] > 1).astype(int)

        cards_per_user = df.groupby("user_id")["card_number"].transform("nunique")
        df["multi_card_user"]    = (cards_per_user >= 2).astype(int)

        # Simple time gap proxy per user
        df = df.sort_values(["user_id", "transaction_date"])
        df["time_diff_user"] = df.groupby("user_id")["transaction_date"].diff().dt.total_seconds()

        # Select numeric features only
        feats = [
            "transaction_amount", "hour",
            "amount_gt_1000", "is_business_hour",
            "tx_per_card", "tx_per_user", "tx_per_device",
            "card_tx_gt1", "user_tx_gt1", "device_tx_gt1",
            "merchants_per_device", "device_merch_gt1",
            "merchants_per_user", "user_merch_gt1",
            "merchants_per_card", "card_merch_gt1",
            "users_per_card", "card_users_gt1",
            "multi_card_user", "time_diff_user"
        ]
        for c in feats:
            if c not in df.columns:
                df[c] = np.nan
        return df[feats]

# Train/validate/test split
df = load_dataframe(csv_path)
y  = df["has_cbk"].values
Xb = df[["transaction_id","merchant_id","user_id","card_number",
         "transaction_date","transaction_amount","device_id"]].copy()

fb = FeatureBuilder()
X_all = fb.transform(Xb)
X_all = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_all), columns=X_all.columns)

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_all, y, test_size=0.4, random_state=42, stratify=y)
X_va, X_te,  y_va, y_te  = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)


# Candidate models (simple configs)
candidates = {
    "LogReg": LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42
    ),
}
candidates["LightGBM"] = LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", n_jobs=-1, verbose=-1, random_state=42
    )


# Evaluate on validation set to pick winner
rows = []
curves = {}

for name, model in candidates.items():
    model.fit(X_tr, y_tr)
    p_va = model.predict_proba(X_va)[:,1]
    yhat_va = (p_va >= 0.5).astype(int)

    auc  = roc_auc_score(y_va, p_va)
    ap   = average_precision_score(y_va, p_va)
    prec = precision_score(y_va, yhat_va, zero_division=0)
    rec  = recall_score(y_va, yhat_va, zero_division=0)
    f1   = f1_score(y_va, yhat_va, zero_division=0)

    fpr, tpr, _ = roc_curve(y_va, p_va)
    pr, rc, _   = precision_recall_curve(y_va, p_va)
    curves[name] = (fpr, tpr, pr, rc)

    rows.append({"model": name, "val_auc": auc, "val_ap": ap,
                 "val_precision@0.5": prec, "val_recall@0.5": rec, "val_f1@0.5": f1})

results = pd.DataFrame(rows).sort_values(["val_auc","val_ap"], ascending=False).reset_index(drop=True)

# Choose winner by highest AUC (break ties with AP)
best_name = results.iloc[0]["model"]
best_model = candidates[best_name]

# Final test evaluation for the winner
p_te = best_model.predict_proba(X_te)[:,1]
yhat_te = (p_te >= 0.5).astype(int)

test_auc = roc_auc_score(y_te, p_te)
test_ap  = average_precision_score(y_te, p_te)
test_prec= precision_score(y_te, yhat_te, zero_division=0)
test_rec = recall_score(y_te, yhat_te, zero_division=0)
test_f1  = f1_score(y_te, yhat_te, zero_division=0)
test_report = classification_report(y_te, yhat_te, digits=3)

# Export findings
# 1. TXT summary with final decision + test metrics
with open(output_dir / "model_selection_report.txt", "w", encoding="utf-8") as f:
    f.write("=== Model Validation Comparison ===\n")
    f.write(results.to_string(index=False))
    f.write("\n\n=== Winner (by AUC then AP) ===\n")
    f.write(f"{best_name}\n")
    f.write("\n=== Test Metrics (Winner) ===\n")
    f.write(f"AUC: {test_auc:.3f}\nAP: {test_ap:.3f}\n")
    f.write(f"Precision@0.5: {test_prec:.3f}\nRecall@0.5: {test_rec:.3f}\nF1@0.5: {test_f1:.3f}\n\n")
    f.write(test_report)

# 2. Plots: ROC + PR for all models (validation set)
plt.figure(figsize=(6,4))
for name, (fpr, tpr, _, _) in curves.items():
    plt.plot(fpr, tpr, label=name)
plt.plot([0,1],[0,1],"--", lw=1, color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Validation)")
plt.legend(); plt.tight_layout()
plt.savefig(output_dir / "chart_val_roc_all.png", dpi=150); plt.close()

plt.figure(figsize=(6,4))
for name, (_, _, pr, rc) in curves.items():
    plt.plot(rc, pr, label=name)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (Validation)")
plt.legend(); plt.tight_layout()
plt.savefig(output_dir / "chart_val_pr_all.png", dpi=150); plt.close()

# 4) Confidence histogram for the winner on test
plt.figure(figsize=(6,4))
plt.hist(p_te, bins=20, edgecolor="black")
plt.xlabel("Predicted probability of chargeback (winner)"); plt.ylabel("Count (test)")
plt.title(f"Confidence Histogram — Test ({best_name})")
plt.tight_layout()
plt.savefig(output_dir / "chart_test_confidence_hist_winner.png", dpi=150); plt.close()

print("Saved:")
print(f"- {output_dir / 'model_selection_report.txt'}")
print(f"- {output_dir / 'chart_val_roc_all.png'}")
print(f"- {output_dir / 'chart_val_pr_all.png'}")
print(f"- {output_dir / 'chart_test_confidence_hist_winner.png'}")
print(f"\nWinner: {best_name} | Test AUC={test_auc:.3f} AP={test_ap:.3f}")