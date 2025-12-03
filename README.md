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

## Repository structure
.
├── data/
│   ├── okm_diak_adat.csv              # raw OKM student-level dataset
│   ├── filtered_data_anal.csv         # 90% training/analysis sample
│   └── filtered_data_eval.csv         # 10% evaluation/holdout sample
├── codes/
│   ├── 01_data_modification.ipynb     # data cleaning, NA handling, feature engineering
│   ├── 02_regressions.ipynb           # regression models and diagnostics
│   ├── 03_tree_forest.ipynb           # decision trees and random forests
│   ├── 04_boosting.ipynb              # boosting models (e.g. gradient boosting)
│   └── 05_main.py                     # main script to run the full modelling pipeline
├── figures/
│   └── na_plot.png                    # visualization of missing values by variable
├── documents/
│   └── analysis.pdf                   # written report in .pdf format
├── requirements.txt                   # Python dependencies
└── README.md                          # project description and usage
