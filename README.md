# Heart Disease Prediction (PRCP-1016)

Machine learning project to predict heart disease from 13 clinical features.

## Dataset
180 patients, 13 clinical features, no missing values, fairly balanced classes (55.6% vs 44.4%).
Source: https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1016-HeartDieseasePred.zip

## Approach
- Exploratory data analysis (distributions, correlations, outliers)
- Preprocessing in a scikit-learn Pipeline (one-hot encoding, scaling)
- Compared 9 models with 5-fold stratified cross-validation
- Tuned Logistic Regression, Random Forest and XGBoost with GridSearchCV
- Evaluated on a held-out test set (36 patients)

## Results
- Cross-validation ROC-AUC after tuning: about 0.88 for the top three models
- Test set (Logistic Regression): recall 0.94, ROC-AUC 0.96 (small test set, so likely optimistic)
- Strongest predictors: thal result, number of major vessels, chest pain type, exercise-induced angina, ST depression, max heart rate

## Limitations
Small dataset, no external validation. This is a learning project, not a clinical tool.

## How to run
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
jupyter notebook

## Files
- `PRCP-1016-HeartDiseasePred.ipynb`: full analysis and models
- `values.csv`, `labels.csv`: dataset
- `heart_disease_model.pkl`: saved Logistic Regression pipeline
