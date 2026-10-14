# Employee Attrition Model Evaluation Report

## Objective and methodology
This report compares four baseline classifiers and a GridSearchCV-tuned Random Forest on the IBM HR Analytics employee dataset. Task 2's fixed stratified 80/20 split (seed 42) is reused. All preprocessing is cloned and fitted within each training fold during cross-validation, then fitted on the full training partition for final test evaluation. The target `Attrition` is never included in model features.

## Dataset and target
The packaged Task 2 cleaned dataset contains 1,470 employee rows and 37 columns, with `Attrition` as the binary target (`No`/`Yes`; Yes is positive). The fixed split contains 1,176 training rows and 294 held-out test rows; test class counts are 247 No and 47 Yes. Task 2 quality checks found zero missing cells and zero exact duplicate rows.

## Algorithms and settings
The four baselines are Logistic Regression (balanced class weight, 2,000 iteration cap), Decision Tree (balanced class weight, minimum leaf size 3), Random Forest (200 trees, balanced subsampling, minimum leaf size 2), and K-Nearest Neighbors (7 neighbors, distance weighting). The tuned Random Forest uses 5-fold stratified GridSearchCV and positive-class F1 scoring across 12 candidates: n_estimators 100/200, max_depth 5/10/None, and min_samples_split 2/5.

## Held-out test comparison
All values below are measured on the same held-out Task 2 test set. Precision, recall and F1 refer to Attrition=Yes.

| Candidate | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.786 | 0.402 | 0.702 | 0.512 | 0.814 |
| Decision Tree | 0.769 | 0.322 | 0.404 | 0.358 | 0.617 |
| Random Forest (baseline) | 0.827 | 0.389 | 0.149 | 0.215 | 0.786 |
| K-Nearest Neighbors | 0.847 | 0.625 | 0.106 | 0.182 | 0.655 |
| Random Forest (GridSearchCV tuned) | 0.833 | 0.480 | 0.511 | 0.495 | 0.792 |

## Grid search and model selection
Training-only cross-validation selected `{'classifier__max_depth': 5, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 200}` with mean CV F1 0.485. The candidate is selected exclusively by highest mean positive-class F1 from training-only stratified cross-validation (the tuned RF uses its GridSearchCV CV score). The selected candidate is **Logistic Regression**. The fixed test partition is reserved for final evaluation only.

## Interpretation and business trade-offs
The target is imbalanced (only 47 of 294 test employees are labeled Yes). Accuracy can therefore hide missed attrition cases. Higher recall catches more potential leavers but may create more false positives and unnecessary outreach; higher precision reduces unnecessary interventions but can miss employees needing support. Confusion matrices and ROC curves provide complementary views. ROC-AUC measures ranking over thresholds and does not identify a practical operating threshold. Correlation and feature importance do not establish causation.

## Limitations, responsible use, and privacy
This dataset is a historical, synthetic-style HR analytics benchmark distributed for research/education, not a representative causal study of a particular employer. The small positive class, potential historical bias, proxy variables, and distribution shift limit generalization. Attrition predictions must be decision-support signals only; never use them for punitive decisions or automated employment actions. Restrict access to personal data, minimize retention, assess subgroup error rates and disparate impact, obtain human review, and provide fair support independent of the score.

## Reproducibility and model persistence
Random seed 42 and the Task 2 split manifest are used. The selected estimator is a Pipeline containing preprocessing and classifier. It was saved and reloaded; reloaded predictions matched the fitted pipeline on all 294 test records, and 5 compatible unseen rows (including an unknown JobRole) transformed and predicted successfully.

## Charts
Confusion matrices are provided for each candidate, with a shared held-out ROC plot and feature importance when supported. See the accompanying image files and captions in the PDF.

## References
- IBM HR Analytics Employee Attrition & Performance dataset, IBM sample dataset commonly distributed via Kaggle: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset (dataset attribution; see Task 2 provenance).
- Scikit-learn documentation: Pipeline, ColumnTransformer, GridSearchCV, classification metrics.
