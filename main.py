import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/ckd-dataset-v2.csv"
OUT = "outputs_ckd"
os.makedirs(OUT, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv(DATA_PATH)
data.replace(["?", "\t?", " \t?", "\t"], np.nan, inplace=True)

target_col = next(c for c in ["classification", "class", "target"] if c in data.columns)
data.rename(columns={target_col: "Outcome"}, inplace=True)

data["Outcome"] = (
    data["Outcome"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"ckd": 1, "notckd": 0})
)

data = data.dropna(subset=["Outcome"])
data["Outcome"] = data["Outcome"].astype(int)

# -----------------------------
# Missing value visualization
# -----------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False)
plt.savefig(f"{OUT}/missing_values.png")
plt.close()

# -----------------------------
# Feature / Target split
# -----------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.dropna(axis=1, how="all")

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

if cat_cols:
    X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
    for c in cat_cols:
        X[c] = LabelEncoder().fit_transform(X[c])

if num_cols:
    X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])

# -----------------------------
# Feature distributions
# -----------------------------
X.hist(figsize=(14, 10))
plt.tight_layout()
plt.savefig(f"{OUT}/feature_distributions.png")
plt.close()

# -----------------------------
# Class distribution
# -----------------------------
plt.figure()
sns.countplot(x=y)
plt.savefig(f"{OUT}/class_before_smote.png")
plt.close()

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# SMOTE
# -----------------------------
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

plt.figure()
sns.countplot(x=y_train)
plt.savefig(f"{OUT}/class_after_smote.png")
plt.close()

# -----------------------------
# Model training
# -----------------------------
model = RandomForestClassifier(
    n_estimators=80,
    max_depth=5,
    min_samples_leaf=10,
    min_samples_split=12,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

with open(f"{OUT}/cv_results.txt", "w") as f:
    f.write(f"Mean Accuracy: {cv_scores.mean()*100:.2f}%\n")
    f.write(f"Std Dev: {cv_scores.std()*100:.2f}%\n")

model.fit(X_train, y_train)

# -----------------------------
# SAVE MODEL + FEATURES (✅ CORRECT PLACE)
# -----------------------------
joblib.dump(model, f"{OUT}/ckd_rf_model.pkl")
joblib.dump(X.columns.tolist(), f"{OUT}/feature_names.pkl")

print("Model and feature names saved successfully.")

# -----------------------------
# Evaluation
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.65).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

with open(f"{OUT}/metrics.txt", "w") as f:
    f.write(
        f"Accuracy: {acc*100:.2f}%\n"
        f"Precision: {prec*100:.2f}%\n"
        f"Recall: {rec*100:.2f}%\n"
        f"F1-score: {f1*100:.2f}%\n"
    )

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.savefig(f"{OUT}/confusion_matrix.png")
plt.close()

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.legend()
plt.savefig(f"{OUT}/roc_curve.png")
plt.close()

# -----------------------------
# Precision-Recall Curve
# -----------------------------
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(recall, precision)
plt.savefig(f"{OUT}/precision_recall_curve.png")
plt.close()

# -----------------------------
# Feature importance
# -----------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.savefig(f"{OUT}/feature_importance.png")
plt.close()

# -----------------------------
# SHAP
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_test)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

plt.figure()
shap.summary_plot(shap_vals, X_test, show=False)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_summary.png")
plt.close()

print("Pipeline completed successfully.")
