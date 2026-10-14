# AI/ML Internship Task 3 - Employee Attrition Classification

This standalone project compares four classification algorithms and tunes a Random Forest using the Task 2 Employee Attrition dataset. The target is `Attrition` (`Yes` is the positive class). Task 2's fixed, stratified 80/20 split (seed 42) is reused. Preprocessing remains inside sklearn Pipelines and is fitted within training folds only. The test partition is reserved for final evaluation.

## Responsible use

This is an educational benchmark, not an employee decision system. Attrition predictions can be biased, reflect historical conditions, and be wrong. Use only as a decision-support signal for voluntary, supportive follow-up, with privacy controls, subgroup fairness checks, and meaningful human review. Do not use for punitive or automated employment actions. Metrics and feature importance do not establish causation.

## Requirements and setup (Windows PowerShell)

Python 3.11+ is recommended (Task 1's environment on this computer is Python 3.13.7). From this project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once for the current terminal: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. In Command Prompt, use `.venv\Scripts\activate.bat`.

## Data and provenance

`data/processed/employee_attrition_clean.csv` and `train_test_split_manifest.json` are copied from Task 2. The source is the IBM HR Analytics Employee Attrition & Performance educational dataset, commonly distributed through [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). The file is a derived cleaned copy; Task 2 reports 1,470 rows, 37 columns, no missing cells or exact duplicate rows. The split manifest retains Task 2's original row indices and class counts. `models/preprocessor_task2.pkl` is the companion Task 2 transformation blueprint; compatible helper classes are included in `src/`.

If you need to recover data, download the IBM HR Analytics Employee Attrition & Performance dataset from the linked dataset page (or the source indicated by your Task 2 submission), select the processed 1,470-row table, and place it at `data/processed/employee_attrition_clean.csv`. The split manifest and Task 2 preprocessor must correspond to the exact same row order and schema; otherwise regenerate compatible Task 2 artifacts first. Do not substitute generated/synthetic rows.

## Run

From the project root, run the reproducible training and reporting workflow:

```powershell
python -m src.train
python -m src.generate_report
pytest -q
```

Open `notebooks/03_model_training.ipynb` in VS Code with the project `.venv` kernel and run all cells top to bottom. Or execute it from PowerShell:

```powershell
python -m jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb --output 03_model_training.executed.ipynb
```

Training creates `models/best_attrition_model.pkl`, `outputs/model_comparison.csv`, `outputs/evaluation_results.json`, classification reports, confusion matrices for each candidate, a shared ROC plot, and feature importance when supported. The report generation command creates a populated Markdown report and PDF from those run outputs. The 12-candidate RF grid uses 5-fold stratified cross-validation and F1 scoring.

## Project structure

```text
data/processed/       Task 2 cleaned dataset and fixed split manifest
models/               compatible Task 2 preprocessor and selected Task 3 model
notebooks/            executable Task 3 training notebook
src/                  validation, inherited Task 2 transforms, training, metrics, charts, report
tests/                automated data, metrics, pipeline and serialization checks
images/                actual confusion matrices, ROC and feature-importance charts
outputs/               generated metrics and classification report JSON/CSV
reports/               generated Markdown and PDF evaluation reports
screenshots/            reserved for genuine application screenshots if required
```

## Evaluation design

Baselines: Logistic Regression, Decision Tree, Random Forest, and KNN. Random Forest GridSearchCV examines `n_estimators` 100/200, `max_depth` 5/10/None, and `min_samples_split` 2/5. Selection uses the highest mean positive-class F1 from training-only stratified CV (RF uses its GridSearchCV score); the fixed test set is not used for model selection. Reports include accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and class-wise classification reports. Because the target is imbalanced, precision/recall trade-offs matter; threshold policy should be decided with stakeholders before any valid deployment.

## Limitations

The benchmark data is not evidence of causal drivers and may not represent another organization or time period. Test metrics are one split estimate, not a guarantee of future performance. Re-check data quality, subgroup errors, fairness, privacy, calibration, drift, and appropriate human oversight before any real-world use.
