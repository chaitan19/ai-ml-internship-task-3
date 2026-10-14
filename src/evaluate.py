"""Leakage-aware metric evaluation helpers."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report)

def evaluate_classifier(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    classes = list(model.classes_)
    positive_index = classes.index("Yes")
    probabilities = model.predict_proba(X_test)[:, positive_index]
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label="Yes", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label="Yes", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label="Yes", zero_division=0)),
        "roc_auc": float(roc_auc_score((y_test == "Yes").astype(int), probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["No", "Yes"]).tolist(),
        "classification_report": classification_report(y_test, predictions, labels=["No", "Yes"],
            target_names=["No", "Yes"], output_dict=True, zero_division=0),
        "y_pred": predictions.tolist(), "y_score": probabilities.tolist(),
    }
