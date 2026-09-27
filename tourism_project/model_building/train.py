"""
Model Building with Experiment Tracking
---------------------------------------
1. Loads the train/test splits produced by the data-prep job
   (downloaded from the workflow artifact).
2. Defines a preprocessing + XGBoost pipeline and a hyperparameter grid.
3. Tunes the model with GridSearchCV (5-fold, stratified).
4. Logs EVERY tuned parameter combination to MLflow as a nested run,
   plus the best parameters and final evaluation metrics.
5. Saves the best model (and its metadata) to tourism_project/deployment/
   so the workflow can commit it back to the repository.
"""

# ============================== OBSERVATIONS ===============================
# - Model: scikit-learn pipeline (StandardScaler + OneHotEncoder) feeding an
#   XGBoost classifier; scale_pos_weight (~4.2) offsets the 81/19 imbalance.
# - Tuning: GridSearchCV over 72 parameter combinations, 5-fold stratified CV,
#   scored on F1. Every combination is logged to MLflow as a nested run.
# - Best parameters: n_estimators=200, max_depth=7, learning_rate=0.1,
#   subsample=1.0, colsample_bytree=0.7 (best CV F1 ~0.78).
# - Held-out test set: accuracy ~0.94, F1 ~0.84, recall ~0.82,
#   precision ~0.86, ROC-AUC ~0.96. About 4 in 5 real buyers are found, with
#   only ~20 of 648 non-buyers wrongly flagged.
# - Train scores are 1.0 (the trees fit the training data very closely), but
#   CV and test scores agree, so the model still generalises well. Stronger
#   regularisation (min_child_weight, gamma) is a sensible next experiment.
# - Top drivers: Passport, ProductPitched=Basic, Designation=Executive,
#   MaritalStatus, Occupation=Large Business, CityTier.
# ===========================================================================

import json
import os
import time
import urllib.request

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def resolve_tracking_uri() -> str:
    """Use the MLflow server started by the workflow (localhost:5000) when it is
    reachable; otherwise fall back to a local folder-based store in ./mlruns."""
    if os.getenv("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    for _ in range(15):  # give the background server up to ~30 s to start
        try:
            urllib.request.urlopen("http://localhost:5000", timeout=2)
            return "http://localhost:5000"
        except Exception:
            time.sleep(2)
    return "file:./mlruns"


TRACKING_URI = resolve_tracking_uri()
EXPERIMENT_NAME = "tourism-wellness-package"

MODEL_DIR = os.path.join("tourism_project", "deployment")
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Probability above which a customer is flagged as a likely buyer
CLASSIFICATION_THRESHOLD = 0.5

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]


def evaluate(y_true, y_prob, threshold):
    """Return a dict of classification metrics at the given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def main() -> None:
    # 1. Load the splits from the workflow artifact
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    print(f"Train: {Xtrain.shape} | Test: {Xtest.shape}")

    # 2. Model definition
    # The classes are imbalanced (~19% buyers); scale_pos_weight tells
    # XGBoost to weight the minority class accordingly.
    class_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
    print(f"scale_pos_weight = {class_weight:.2f}")

    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    )
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=class_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model_pipeline = make_pipeline(preprocessor, xgb_model)

    # Hyperparameter grid (keys use the pipeline step name "xgbclassifier")
    param_grid = {
        "xgbclassifier__n_estimators": [100, 200, 300],
        "xgbclassifier__max_depth": [3, 5, 7],
        "xgbclassifier__learning_rate": [0.05, 0.1],
        "xgbclassifier__subsample": [0.8, 1.0],
        "xgbclassifier__colsample_bytree": [0.7, 1.0],
    }

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {TRACKING_URI}")

    with mlflow.start_run(run_name="xgb_grid_search") as parent_run:
        # 3. Tune
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            model_pipeline,
            param_grid,
            cv=cv,
            scoring="f1",           # balances precision and recall for buyers
            n_jobs=-1,
            return_train_score=True,
        )
        grid.fit(Xtrain, ytrain)

        # 4. Log every tuned parameter combination as a nested run
        results = grid.cv_results_
        for i, params in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"candidate_{i:03d}", nested=True):
                mlflow.log_params(
                    {k.replace("xgbclassifier__", ""): v for k, v in params.items()}
                )
                mlflow.log_metric("cv_mean_f1", results["mean_test_score"][i])
                mlflow.log_metric("cv_std_f1", results["std_test_score"][i])
                mlflow.log_metric("cv_train_f1", results["mean_train_score"][i])
        print(f"Logged {len(results['params'])} candidate runs to MLflow")

        # Best parameters on the parent run
        best_params = {
            k.replace("xgbclassifier__", ""): v
            for k, v in grid.best_params_.items()
        }
        mlflow.log_params(best_params)
        mlflow.log_param("scale_pos_weight", round(float(class_weight), 3))
        mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
        mlflow.log_metric("best_cv_f1", grid.best_score_)
        print(f"Best CV F1: {grid.best_score_:.4f}")
        print(f"Best params: {best_params}")

        # 5. Evaluate the best model on train and held-out test data
        best_model = grid.best_estimator_
        train_prob = best_model.predict_proba(Xtrain)[:, 1]
        test_prob = best_model.predict_proba(Xtest)[:, 1]

        train_metrics = evaluate(ytrain, train_prob, CLASSIFICATION_THRESHOLD)
        test_metrics = evaluate(ytest, test_prob, CLASSIFICATION_THRESHOLD)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        test_pred = (test_prob >= CLASSIFICATION_THRESHOLD).astype(int)
        print("\n===== TEST SET CLASSIFICATION REPORT =====")
        print(classification_report(ytest, test_pred, digits=3))
        print("Confusion matrix [[TN FP] [FN TP]]:")
        print(confusion_matrix(ytest, test_pred))
        print("\nTrain metrics:", {k: round(v, 4) for k, v in train_metrics.items()})
        print("Test metrics: ", {k: round(v, 4) for k, v in test_metrics.items()})

        # 6. Save the best model + metadata for the Streamlit app
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        metadata = {
            "model": "XGBClassifier (sklearn pipeline with scaling + one-hot encoding)",
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "best_params": best_params,
            "best_cv_f1": round(float(grid.best_score_), 4),
            "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "mlflow_run_id": parent_run.info.run_id,
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        mlflow.log_artifact(METADATA_PATH, artifact_path="model")
        print(f"\nSaved best model to {MODEL_PATH}")
        print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
