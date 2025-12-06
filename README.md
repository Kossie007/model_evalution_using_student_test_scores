# Predicting Student Performance from Competency Assessments
## A Comparison of Tree-Based Models on Raw and MissForest-Imputed Data

### Course: Applying and Interpreting Machine Learning – Group Project

### Team Members:
-  Kristóf Andrási
-  Gellért Banai
-  Hunor Kuti
-  Kristóf Légrádi
-  Ákos Virág

### Project Overview
This repository contains the code and analysis for a group project investigating the predictors of student academic performance in Hungary. We analyze primary school literature and mathematics scores from the 2019 National Assessment of Basic Competencies (Országos Kompetenciamérés - OKM), covering a dataset of over 70,000 students.

The primary objective of this project is to predict student scores using various socio-economic and school-level features. A specific focus is placed on the methodological comparison between using tree-based machine learning models on raw data versus data imputed using the MissForest algorithm.

### Repository Structure
```text
.
├── data/
│   ├── okm_diak_adat.csv              # raw OKM student-level dataset
│   └── filtered_data_anal.csv         # cleaned data for analysis
├── codes/
│   ├── 01_data_modification.ipynb     # data cleaning, NA handling, feature engineering
│   ├── 02_regressions.ipynb           # regression models and diagnostics
│   ├── 03_tree_forest.ipynb           # decision trees and random forests
│   ├── 04_boosting.ipynb              # boosting models
│   └── 05_main.py                     # main script to run the full modelling pipeline
├── figures/
│   └── na_plot.png                    # visualization of missing values by variable
├── documents/
│   └── analysis.pdf                   # written report in .pdf format
├── requirements.txt                   # Python dependencies
└── README.md                          # project description and usage
```

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

file: `05_main.py`
-  A consolidated script designed to run the full modeling pipeline from end to end.

#### Data
The data/ folder contains the datasets derived from the 2019 Hungarian National Assessment of Basic Competencies.

file: `okm_diak_adat.csv`
-  The raw, student-level dataset containing math and literature scores, along with background questionnaires regarding socio-economic status and school environment.

file: `filtered_data_anal.csv`
-  The training and analysis set, comprising 100% of the filtered data. This is split for training and test sets during the analysis.


file: `variable_description.txt`
-  Introductory dataset of used variables giving ranges, basic statistics and categories.


#### Figures and Reports

file: `figures/na_plot.png`
-  Visualizes the distribution and frequency of missing values across variables, justifying the need for advanced imputation strategies like MissForest.

file: `documents/analysis.pdf`
-  The final written report summarizing the methodology, model comparisons, interpretation of feature importance, and conclusions regarding the efficacy of imputation in educational data mining.

```text
├── figures/
│   └── na_plot.png                    # visualization of missing values by variable
├── documents/
│   └── analysis.pdf                   # written report in .pdf format
├── requirements.txt                   # Python dependencies
└── README.md                          # project description and usage
```

### Licence
MIT License (MIT): see the [License File](https://github.com/sensiolabs/GotenbergBundle/blob/1.x/LICENSE) for more details.

