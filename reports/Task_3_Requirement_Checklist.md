# Task 3 acceptance checklist

| Requirement | Status | Evidence |
|---|---|---|
| IBM HR Attrition data used | PASS | Packaged Task 2 cleaned CSV (1,470 x 37), dataset attribution in README/report. |
| Reuse Task 2 split and target separate | PASS | 1,176/294 split manifest, seed 42, stratified; `Attrition` omitted from X. |
| Data inspection and quality checks | PASS | Notebook prints shape, columns, dtypes, first ten rows, descriptive stats, missing cells, duplicates, class counts. |
| Four baseline classifiers | PASS | Logistic Regression, Decision Tree, Random Forest, KNN trained in `src/train.py`. |
| Required evaluation metrics and classification reports | PASS | Accuracy, positive-class precision/recall/F1, ROC-AUC, confusion matrix, classification report for all five candidates in generated outputs. |
| Confusion matrices and ROC curve | PASS | Five held-out confusion matrix PNGs and combined ROC PNG embedded in report. |
| RF GridSearchCV | PASS | 5-fold stratified training-only CV, 12 combinations across tree count/depth/split size; F1 for Attrition=Yes. |
| Leakage prevention | PASS | Task 2 split reused; preprocessors cloned inside pipeline and fitted per CV training fold; test used only for final evaluation. |
| Model selection rule | PASS | Highest mean Attrition=Yes F1 from training-only CV; test data does not select candidates. |
| Model saved and reloaded | PASS | `best_attrition_model.pkl`; reloaded predictions match original on 294 held-out rows. |
| Compatible unseen data | PASS | Five test-like rows transformed/predicted, including unknown JobRole value. |
| Notebook complete and executed | PASS | Executed notebook saved with outputs; no cell errors. |
| PDF report and Markdown report | PASS | Generated from actual run artifacts; includes charts, metrics, tuning, decision trade-offs and responsible-use notes. |
| Automated tests | PASS | `4 passed`; includes schema/split, preprocessing clone, metrics, training and persistence/unseen test. |
| README and requirements | PASS | Windows setup, commands, data provenance, outputs, limitations documented. |
| Execution screenshots | PARTIAL | No genuine application-window screenshots were captured. Visualizations and executed notebook outputs are present; capture actual VS Code/Jupyter screenshots manually if explicitly required for submission. |
| Separate Task 3 Git repository | PASS | Initialized within the dedicated Task 3 folder only; Task 1/Task 2 untouched. |
| GitHub commit/push | BLOCKED | New repository HTTPS/SSH URL has not been provided. No remote was guessed and no push occurred. Provide the URL to continue. |
