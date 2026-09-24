# Importing the dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder  # What is label encoder ?
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,  # What is the use of this ?
)

# from imblearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,  # What this the use of this ?
    classification_report,
)

# LOADING DATA AND INITIAL VIEWING OF DATA

df = pd.read_csv("./dataset/churn_data.csv")

# Check number of rows and cloumns ( 7043 rows, 21 coulmns)
print(df.shape)

# Get first 5 items of the csv

print(df.head())

print(df.info())

NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "SeniorCitizen"]

# Categorial Columns
for col in df.columns:
    if col not in NUMERICAL_COLUMNS:
        print(col, df[col].unique())

# DATA CLEANING
# Dropping customerId Column as this si not required for modeling
df = df.drop(columns=("customerID"))
print(df.head(2))

# Change the datatype of TotalCharges from string to float

# df["TotalCharges"] = df["TotalCharges"].astype(float)  # This will crash

# Filter rows in TotalCharges that are ' ' or " " ( contain empty space )

print(df[df["TotalCharges"] == " "])
print(len(df[df["TotalCharges"] == " "]))  # 11
# """
# python3 main.py
#       customerID  gender  SeniorCitizen Partner Dependents  tenure  ...  Contract PaperlessBilling              PaymentMethod MonthlyCharges TotalCharges Churn
# 488   4472-LVYGI  Female              0     Yes        Yes       0  ...  Two year              Yes  Bank transfer (automatic)          52.55                 No
# 753   3115-CZMZD    Male              0      No        Yes       0  ...  Two year               No               Mailed check          20.25                 No
# 936   5709-LVOEQ  Female              0     Yes        Yes       0  ...  Two year               No               Mailed check          80.85                 No
# 1082  4367-NUYAO    Male              0     Yes        Yes       0  ...  Two year               No               Mailed check          25.75                 No
# 1340  1371-DWPAZ  Female              0     Yes        Yes       0  ...  Two year               No    Credit card (automatic)          56.05                 No
# 3331  7644-OMVMY    Male              0     Yes        Yes       0  ...  Two year               No               Mailed check          19.85                 No
# 3826  3213-VVOLG    Male              0     Yes        Yes       0  ...  Two year               No               Mailed check          25.35                 No
# 4380  2520-SGTTA  Female              0     Yes        Yes       0  ...  Two year               No               Mailed check          20.00                 No
# 5218  2923-ARZLG    Male              0     Yes        Yes       0  ...  One year              Yes               Mailed check          19.70                 No
# 6670  4075-WKNIU  Female              0     Yes        Yes       0  ...  Two year               No               Mailed check          73.35                 No
# 6754  2775-SEFEE    Male              0      No        Yes       0  ...  Two year              Yes  Bank transfer (automatic)          61.90                 No

# [11 rows x 21 columns]
# """ Contain empty strings for null values

# Cleaning TotalCharges

df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)
print(len(df[df["TotalCharges"] == " "]))

# Understanding the distribution of target column ! IMPORTANT target = Churn
print(df["Churn"].value_counts())
# """
# Churn
# No     5174
# Yes    1869
# Name: count, dtype: int64
# """

# EXPLAINING DATA ANALYSIS
