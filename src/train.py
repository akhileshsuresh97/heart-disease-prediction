"""Train the heart disease models and save them to models/.

Usage:
    python src/train.py

Steps:
1. Load the UCI Cleveland dataset (303 patients) and binarise the target
   (num > 0 -> heart disease).
2. Compare six classifiers with GridSearchCV (5-fold stratified CV, ROC AUC).
   Imputation and scaling live inside each pipeline, so nothing leaks from
   the test set into training.
3. Evaluate the best model on a held-out 20% test set.
4. Train a secondary model that flags rule-based risk profiles
   (hypertension / diabetes indicators) for patients without heart disease.
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed.cleveland.data"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
SEED = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, header=None, names=FEATURES + ["num"], na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    return df.drop(columns="num")


def make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


CANDIDATES = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10],
         "model__min_samples_split": [2, 5]},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1],
         "model__max_depth": [2, 3]},
    ),
    "SVC": (
        CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False),
        {"model__estimator__kernel": ["linear", "rbf"], "model__estimator__C": [0.1, 1, 10]},
    ),
    "K-Nearest Neighbors": (
        KNeighborsClassifier(),
        {"model__n_neighbors": [5, 7, 9, 11], "model__weights": ["uniform", "distance"]},
    ),
    "Extra Trees": (
        ExtraTreesClassifier(random_state=SEED),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10]},
    ),
}


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def secondary_labels(df: pd.DataFrame) -> pd.Series:
    """Rule-based labels from the original project (not clinical diagnoses).

    1 = hypertension indicators: age > 50, cholesterol > 200, resting BP > 140
    2 = diabetes indicators:     fasting blood sugar > 120 and cholesterol > 200
    0 = neither
    """
    htn = (df["age"] > 50) & (df["chol"] > 200) & (df["trestbps"] > 140)
    dm = (df["fbs"] == 1) & (df["chol"] > 200)
    return pd.Series(np.select([htn, dm], [1, 2], default=0), index=df.index)


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    df = load_data()
    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1. Model comparison
    results, fitted = [], {}
    for name, (model, grid) in CANDIDATES.items():
        search = GridSearchCV(make_pipeline(model), grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        row = {"model": name, "cv_roc_auc": search.best_score_,
               **evaluate(search.best_estimator_, X_test, y_test)}
        results.append(row)
        print(f"{name:20s} CV AUC {row['cv_roc_auc']:.3f} | test AUC {row['roc_auc']:.3f} "
              f"| acc {row['accuracy']:.3f}")

    results_df = pd.DataFrame(results).sort_values("cv_roc_auc", ascending=False)
    # Select on cross-validation score, not the test set, so the test score stays honest
    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    print(f"\nBest model (by CV ROC AUC): {best_name}")

    # 2. Figures
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = results_df.sort_values("roc_auc")
    ax.barh(plot_df["model"], plot_df["roc_auc"], color="#b3263a")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Test ROC AUC")
    ax.set_title("Model comparison (UCI Cleveland, 20% held-out test set)")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test, display_labels=["No disease", "Disease"],
        cmap="Reds", ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion matrix: {best_name}")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=150)

    inner = best_model.named_steps["model"]
    if hasattr(inner, "feature_importances_"):
        imp = pd.Series(inner.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(imp.index, imp.values, color="#1d3557")
        ax.set_title(f"Feature importance: {best_name}")
        fig.tight_layout()
        fig.savefig(FIGURES / "feature_importance.png", dpi=150)

    # 3. Secondary model. The app only uses it when no heart disease is predicted,
    # but the rule labels apply to every patient, so it learns from all 303.
    y_sec = secondary_labels(df)
    # A decision tree learns axis-aligned thresholds, which is exactly what the rules are
    sec_model = make_pipeline(DecisionTreeClassifier(max_depth=6, random_state=SEED))
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        df[FEATURES], y_sec, test_size=0.25, stratify=y_sec, random_state=SEED
    )
    sec_model.fit(Xs_tr, ys_tr)
    sec_acc = accuracy_score(ys_te, sec_model.predict(Xs_te))
    print(f"Secondary model accuracy (reproducing rule labels): {sec_acc:.3f}")
    sec_model.fit(df[FEATURES], y_sec)  # refit on all patients for deployment

    # 4. Refit the primary model on all data for deployment and save everything
    best_model.fit(X, y)
    joblib.dump(best_model, MODELS / "heart_model.joblib")
    joblib.dump(sec_model, MODELS / "secondary_model.joblib")

    metrics = {
        "dataset": "UCI Heart Disease (Cleveland), 303 patients",
        "test_size": len(X_test),
        "best_model": best_name,
        "comparison": results_df.round(4).to_dict(orient="records"),
        "secondary_model_accuracy": round(sec_acc, 4),
        "secondary_class_counts": y_sec.value_counts().sort_index().to_dict(),
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(results_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
