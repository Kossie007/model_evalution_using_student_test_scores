# Predicting Student Performance from Competency Assessments
## A Comparison of Tree-Based Models on Raw and MissForest-Imputed Data

### Course: Applying and Interpreting Machine Learning – Group Project

### Team Members:
-  Kristóf Andrási
-  Gellért Banai
-  Hunor Kuti
-  Kristóf Légrádi
-  Ákos Virág

---
### Project Overview
This repository contains the code and analysis for a group project investigating the predictors of student academic performance in Hungary. We analyze primary school literature and mathematics scores from the 2019 National Assessment of Basic Competencies (Országos Kompetenciamérés - OKM), covering a dataset of over 70,000 students.

The primary objective of this project is to predict student scores using various socio-economic and school-level features. A specific focus is placed on the methodological comparison between using tree-based machine learning models on raw data versus data imputed using the MissForest algorithm.

---
### Repository Structure
```text
.
├── data/
│   ├── okm_diak_adat.csv              # raw OKM student-level dataset
│   ├── alt_2019_codebook.txt          # raw description of variables in the dataset
│   └── filtered_data_anal.csv         # cleaned data for analysis
├── codes/
│   ├── 01_data_modification.ipynb     # data cleaning, NA handling, feature engineering
│   ├── 02_regressions.ipynb           # regression models and diagnostics
│   ├── 03_tree_forest.ipynb           # decision trees and random forests
│   ├── 04_boosting.ipynb              # boosting models
│   ├── 01-04_model_evaluation_using_student_scores.ipynb         # combined python notebook consisting of a synthesized code. runtime ~ 1200 sec.
│   └── 01-04_model_evaluation_using_student_scores.py            # combined Python file consisting of a synthesized code. runtime ~ 1200 sec.
├── documents/
│   └── analysis.pdf                   # written report in .pdf format
├── figures/
│   ├── all_na_plot.png                # Missing values by variable in the raw, unfiltered dataset
│   ├── decision_tree_tree.png         # Visual representation of the best decision tree regressor
│   ├── distribution_of_math_score.png # Histogram of standardized 8th-grade math scores
│   ├── feature_importance.png         # Top feature importances from the Random Forest model
│   ├── filtered_na_plot.png           # Missing values by variable after basic filtering, before final cleaning
│   ├── lasso1_plot.png                # Lasso: Test RMSE vs. lambda (wide alpha grid)
│   ├── lasso2_plot.png                # Lasso: Test RMSE vs. lambda (narrow alpha grid)
│   ├── residuals_histogram.png        # Histogram of model residuals (LR baseline)
│   ├── residuals_vs_fitted.png        # Residuals vs. fitted values (LR baseline)
│   ├── ridge_plot.png                 # Ridge: Test RMSE vs. lambda
│   ├── start_data_plot.png            # Missing values by variable in the final cleaned analysis dataset
│   ├── true_vs_predicted.png          # True vs. predicted math scores (non-XGB baseline model)
│   ├── xgb_feature_importance.png     # Top feature importances from the tuned XGBoost model
│   ├── xgb_residuals_histogram.png    # Histogram of residuals from the XGBoost model
│   ├── xgb_residuals_vs_fitted.png    # Residuals vs. fitted values for the XGBoost model
│   └── xgb_true_vs_predicted.png      # True vs. predicted math scores for the XGBoost model
└── README.md                          # project description and usage
```

---
#### Scripts
The analysis is divided into sequential notebooks and a main execution script located in the codes/ folder.

file: `01_data_modification.ipynb`
-  This notebook handles the initial data cleaning, feature engineering, and splitting. It is specifically responsible for implementing the MissForest algorithm to handle missing values, generating the datasets used in subsequent modeling.
 
file: `02_regressions.ipynb`
-  Contains baseline linear regression models and diagnostic plots to establish a performance benchmark for the more complex machine learning models.

file: `03_tree_forest.ipynb`
-  Implements and tunes Decision Tree and Random Forest models. This script compares performance metrics across both the raw and the imputed datasets.

file: `04_boosting.ipynb`
-  Implements gradient boosting techniques (e.g., XGBoost/LightGBM) to push predictive performance further, continuing the comparison between data handling strategies.

file: `01-04_model_evaluation_using_student_scores.ipynb`
-  A consolidated script designed to run the full modeling pipeline from end to end.

file: `01-04_model_evaluation_using_student_scores.py`
-  A consolidated script designed to run the full modeling pipeline from end to end.

#### Data
The data/ folder contains the datasets derived from the 2019 Hungarian National Assessment of Basic Competencies.

file: `okm_diak_adat.csv`
-  The raw, student-level dataset containing math and literature scores, along with background questionnaires regarding socio-economic status and school environment.

file: `filtered_data_anal.csv`
-  The training and analysis set, comprising 100% of the filtered data. This is split into training and test sets for analysis.

file: `alt_2019_codebook.txt`
-  Introductory dataset of used variables giving ranges, basic statistics, and categories.


#### Figures and Reports
file: `figures/all_na_plot.png`  
-  Visualizes the distribution and frequency of missing values across all variables in the raw, unfiltered dataset.  
-  Used to justify early decisions on variable removal and the need for further filtering and cleaning.

file: `figures/decision_tree_tree.png`  
-  Plot of the final, tuned decision tree regressor including splits, thresholds, and leaf predictions.  
-  Helps interpret how the tree partitions students into groups and which predictors drive the main splits.

file: `figures/distribution_of_math_score.png`  
-  Histogram of the standardized 8th-grade math score (`math_score_8_std`).  
-  Used to check approximate normality and to confirm that the scaling and standardization behaves as expected.

file: `figures/feature_importance.png`  
-  Bar chart of the top predictor importances from the Random Forest model.  
-  Summarizes which variables contribute most to predictive performance in the ensemble tree model.

file: `figures/filtered_na_plot.png`  
-  Missing-value profile after initial filtering (e.g. dropping high-NA columns and key-variable NAs), but before the final strict cleaning.  
-  Documents which variables are still problematic and informs decisions on further NA handling.

file: `figures/lasso1_plot.png`  
-  Test RMSE versus regularization strength (lambda/alpha) for Lasso over a narrow grid of alpha values.  
-  Used to refine the optimal penalty region once the broad range has been explored and to support model choice.

file: `figures/lasso2_plot.png`  
-  Test RMSE versus lambda/alpha for Lasso over a wide grid of values.  
-  Shows how model performance changes across several orders of magnitude of regularization and helps locate a sensible alpha range.

file: `figures/residuals_histogram.png`  
-  Histogram of residuals (true minus predicted) from a baseline model (e.g. linear or tree-based, non-XGB).  
-  Used to visually check error symmetry, heavy tails, and potential systematic underfile: or over-prediction.

file: `figures/residuals_vs_fitted.png`  
-  Scatter plot of residuals versus fitted (predicted) values for the baseline model.  
-  Helps diagnose heteroskedasticity, non-linearity, and structural misspecification (e.g. patterns or funnels in the residuals).

file: `figures/ridge_plot.png`  
-  Test RMSE as a function of lambda/alpha for Ridge regression.  
-  Supports the choice of regularization strength by showing where error is minimized and how sensitive Ridge is to lambda.

file: `figures/start_data_plot.png`  
-  Missing-value plot for the final cleaned dataset that is actually used for modelling.  
-  Serves as a sanity check that NA-related issues have been resolved before any model fitting.

file: `figures/true_vs_predicted.png`  
-  Scatter plot of true versus predicted math scores for a baseline model (e.g. linear or Random Forest).  
-  If points lie close to the diagonal, the model fits well; systematic deviations indicate bias or lack of fit.

file: `figures/xgb_feature_importance.png`  
-  Horizontal bar chart of the top 20 feature importance ratios from the tuned XGBoost model.  
-  Shows which predictors XGBoost relies on most, providing a non-linear, tree-ensemble view of variable relevance.

file: `figures/xgb_residuals_histogram.png`  
-  Histogram of residuals from the XGBoost model.  
-  Used to compare error distribution against the baseline models and to assess whether XGBoost reduces extreme errors.

file: `figures/xgb_residuals_vs_fitted.png`  
-  Residuals versus predicted values for the XGBoost model.  
-  Checks whether non-linear modeling reduced structural patterns in residuals or if some systematic bias remains.

file: `figures/xgb_true_vs_predicted.png`  
-  Scatter plot of true versus predicted math scores from XGBoost.  
-  Direct visual comparison of XGBoost fit quality; can be contrasted with `true_vs_predicted.png` to see improvement (or not) over simpler models.

file: `documents/analysis.pdf`
-  The final written report summarizes the methodology, model comparisons, interpretation of feature importance, and conclusions regarding the efficacy of imputation in educational data mining.

---
### Licence
MIT License (MIT): see the [License File](https://github.com/sensiolabs/GotenbergBundle/blob/1.x/LICENSE) for more details.

