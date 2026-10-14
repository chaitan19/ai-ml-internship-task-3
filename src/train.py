"""Train baseline attrition classifiers, tune Random Forest, and save the winner."""
from __future__ import annotations
import json, logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from src.utils import ROOT, load_dataset, load_task2_split, load_preprocessor_blueprint
from src.evaluate import evaluate_classifier
from src.visualization import create_model_charts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
RANDOM_STATE = 42

def make_pipeline(preprocessor, classifier):
    return Pipeline([("preprocessor", clone(preprocessor)), ("classifier", classifier)])

def train_and_evaluate(root: Path = ROOT, cv_folds: int = 5):
    data = load_dataset(root / "data/processed/employee_attrition_clean.csv")
    train, test, manifest = load_task2_split(data, root / "data/processed/train_test_split_manifest.json")
    X_train, y_train = train.drop(columns="Attrition"), train["Attrition"].astype(str)
    X_test, y_test = test.drop(columns="Attrition"), test["Attrition"].astype(str)
    if "Attrition" in X_train.columns: raise ValueError("Target leakage: Attrition appears among features.")
    preprocessor = load_preprocessor_blueprint(root / "models/preprocessor_task2.pkl")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    f1_yes = make_scorer(f1_score, pos_label="Yes", zero_division=0)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced", min_samples_leaf=3),
        "Random Forest (baseline)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced_subsample", min_samples_leaf=2),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    }
    fitted, results, selection_cv = {}, [], {}
    for name, estimator in models.items():
        model = make_pipeline(preprocessor, estimator)
        scores = cross_val_score(model, X_train, y_train, scoring=f1_yes, cv=cv, n_jobs=1, error_score="raise")
        selection_cv[name] = float(scores.mean())
        model.fit(X_train, y_train)
        fitted[name] = model
        results.append(evaluate_classifier(name, model, X_test, y_test))
        logging.info("Evaluated %s (training CV F1 %.3f)", name, scores.mean())
    search_pipeline = make_pipeline(preprocessor, RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced_subsample"))
    param_grid = {"classifier__n_estimators": [100, 200], "classifier__max_depth": [5, 10, None], "classifier__min_samples_split": [2, 5]}
    search = GridSearchCV(search_pipeline, param_grid, scoring=f1_yes, cv=cv, n_jobs=1, refit=True, return_train_score=True, error_score="raise")
    search.fit(X_train, y_train)
    fitted["Random Forest (GridSearchCV tuned)"] = search.best_estimator_
    results.append(evaluate_classifier("Random Forest (GridSearchCV tuned)", search.best_estimator_, X_test, y_test))
    # Select exclusively using training-only cross-validation; test is evaluation-only.
    selection_cv["Random Forest (GridSearchCV tuned)"] = float(search.best_score_)
    winner_name = max(selection_cv, key=lambda name: (selection_cv[name], name))
    winner = fitted[winner_name]
    winner_result = next(item for item in results if item["model"] == winner_name)
    model_path = root / "models/best_attrition_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(winner, model_path)
    reloaded = joblib.load(model_path)
    reloaded_pred = reloaded.predict(X_test)
    if not np.array_equal(reloaded_pred, winner.predict(X_test)):
        raise RuntimeError("Reloaded model predictions do not match the fitted model.")
    unseen = X_test.iloc[:5].copy()
    if "JobRole" in unseen: unseen.loc[unseen.index[0], "JobRole"] = "Previously unseen role"
    unseen_predictions = reloaded.predict(unseen)
    if len(unseen_predictions) != len(unseen): raise RuntimeError("Unseen record prediction count mismatch.")
    out = root / "outputs"; out.mkdir(exist_ok=True)
    clean_results = [{k:v for k,v in r.items() if k not in {"y_pred", "y_score"}} for r in results]
    table = pd.DataFrame([{k:r[k] for k in ["model","accuracy","precision","recall","f1","roc_auc"]} for r in results])
    table.to_csv(out / "model_comparison.csv", index=False)
    (out / "evaluation_results.json").write_text(json.dumps({"selection_rule":"Highest mean positive-class F1 on training-only stratified cross-validation. GridSearchCV chooses tuned RF using the same training folds. The fixed test partition is reserved for final evaluation and is not used for selection.", "selection_cv_f1":selection_cv, "winner":winner_name,"task2_split":{"random_state":manifest["random_state"],"test_size":manifest["test_size"],"train_rows":len(train),"test_rows":len(test)},"cv":{"folds":cv_folds,"scoring":"F1 for Attrition=Yes","best_params":search.best_params_,"best_mean_cv_f1":float(search.best_score_),"grid_candidates":len(search.cv_results_["params"])},"results":clean_results,"reload_verified":True,"unseen_rows_verified":len(unseen)}, indent=2), encoding="utf-8")
    (out / "classification_reports.json").write_text(json.dumps({r["model"]:r["classification_report"] for r in results}, indent=2), encoding="utf-8")
    create_model_charts(results, y_test, root / "images")
    # Feature importances for the selected RF when applicable.
    importance_model = fitted["Random Forest (GridSearchCV tuned)"]
    classifier = importance_model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        names = importance_model.named_steps["preprocessor"].get_feature_names_out()
        imp = pd.DataFrame({"feature":names, "importance":classifier.feature_importances_}).sort_values("importance", ascending=False).head(15)
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,5)); imp.sort_values("importance").plot.barh(x="feature", y="importance", legend=False, ax=ax, color="#5671C8")
        ax.set(title=f"Top feature importances - {winner_name}", xlabel="Impurity-based importance", ylabel="Preprocessed feature")
        fig.tight_layout(); fig.savefig(root / "images/feature_importance.png", dpi=160); plt.close(fig)
        imp.to_csv(out / "feature_importance.csv", index=False)
    logging.info("Winner: %s; F1 %.3f, recall %.3f, ROC-AUC %.3f", winner_name, winner_result["f1"], winner_result["recall"], winner_result["roc_auc"])
    return {"winner":winner_name,"results":results,"table":table,"model_path":model_path,"search":search,"reload_verified":True,"unseen_rows_verified":len(unseen),"model":reloaded,"train_rows":len(train),"test_rows":len(test)}

if __name__ == "__main__":
    summary = train_and_evaluate()
    print(summary["table"].to_string(index=False))
    print(f"Best model: {summary['winner']} | saved: {summary['model_path']}")
