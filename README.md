# Applying and Interpreting Machine Learning – Group Project  

## Predicting Student Performance from Competency Assessments: A Comparison of Tree-Based Models on Raw and MissForest-Imputed Data 

### Team members
- Kristóf Andrási  
- Gellért Banai
- Hunor Kuti
- Kristóf Légrádi
- Ákos Virág

## Project overview
We will carry out prediction based analysis on primary school literature and math scores from 2019 including more than 70K students from Hungary. 

.
├── codes/
│   ├── 01_data_modification.ipynb   # data cleaning, imputation, feature engineering
│   ├── 02_regressions.ipynb         # linear / logistic regressions and diagnostics
│   ├── 03_tree_forest.ipynb         # decision trees and random forests
│   ├── 04_boosting.ipynb            # boosting models (e.g. XGBoost, GBM)
│   └── 05_main.py                   # main Python script to run the full pipeline
├── data/
│   ├── okm_diak_adat.csv            # raw student-level OKM dataset
│   ├── filtered_data_anal.csv       # 90% sample for model training / analysis
│   └── filtered_data_eval.csv       # 10% sample for model evaluation
├── figures/
│   └── na_plot.png                  # plot of missing values per variable
├── documents/
│   └── analysis.pdf                 # written report / analysis
├── README.md                        # project description, usage, notes
└── requirements.txt                 # Python dependencies for the project
