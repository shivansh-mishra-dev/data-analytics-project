"""
Customer Churn Prediction — Decision Tree (Telco Churn Dataset)
==================================================================

This script walks through the full modeling process in the order it was
actually done: load -> inspect -> preprocess -> train -> evaluate -> confirm.

Each stage is written as DECISION / WHY / RESULT so the reasoning is visible
alongside the code, not just the code itself. Run this top to bottom; every
print() is a checkpoint meant to be read, not just executed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).parent / "dataset" / "telco_customer_churn.csv"
TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.20

PARAM_GRID = {
    "model__max_depth": [3, 4, 5, 6, 8, 10],
    "model__min_samples_leaf": [1, 10, 25, 50],
    "model__criterion": ["gini", "entropy"],
}


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ======================================================================
# 1. LOAD & FIRST LOOK
# ======================================================================
#
# DECISION: read the raw CSV straight into a DataFrame with no cleaning yet.
# WHY: I want to see the data in its original state before deciding what
#      needs fixing — cleaning blind risks silently hiding real problems.

section("1. LOAD & FIRST LOOK")

data = pd.read_csv(DATA_PATH)

print(f"Shape: {data.shape}")  # RESULT: 7043 rows, 21 columns
print("\nColumn dtypes:")
print(data.dtypes)
print("\nFirst 3 rows:")
print(data.head(3))

# RESULT (observed): TotalCharges is read as an 'object' column, not a
# number — that's a red flag worth chasing in the inspection step below.


# ======================================================================
# 2. INSPECTION — HUNTING FOR PROBLEMS
# ======================================================================
#
# DECISION: check for nulls, look at TotalCharges specifically, and check
#           the target's class balance, before writing any fixes.
# WHY: you can't justify a preprocessing choice you haven't diagnosed —
#      this section is the evidence, the next section is the response.

section("2. INSPECTION")

print("Null counts (pandas' own detector):")
print(data.isnull().sum().sum(), "total nulls found by pandas")
# RESULT: pandas reports 0 nulls — but that's misleading, see below.

blank_total_charges = data[data["TotalCharges"].astype(str).str.strip() == ""]
print(
    f"\nRows where TotalCharges is a blank/whitespace string: {len(blank_total_charges)}"
)
print("Their tenure values:", blank_total_charges["tenure"].unique())
# RESULT: 11 rows have TotalCharges as whitespace, not NaN — pandas missed
# them because a whitespace string isn't technically "null". All 11 have
# tenure == 0, i.e. brand-new customers who haven't been billed yet. This
# is a real, explainable pattern, not random missingness.

print("\nTarget class balance:")
print(data[TARGET].value_counts(normalize=True).round(3))
# RESULT: ~73.5% No / ~26.5% Yes. This imbalance has to shape both the
# model (class weighting) and the evaluation metric (accuracy alone would
# be misleading) — flagged here, acted on in sections 3 and 5.


# ======================================================================
# 3. PREPROCESSING — DECISION, WHY, RESULT FOR EACH STEP
# ======================================================================

section("3. PREPROCESSING")

# --- 3a. Fix TotalCharges -------------------------------------------
# DECISION: convert TotalCharges to numeric, coercing the 11 blank rows
#           to NaN, then let the imputer fill them with 0 rather than a
#           median.
# WHY: the blanks aren't missing-at-random — they're customers with
#      tenure = 0, so they genuinely have not been charged anything yet.
#      A population median would inject a plausible-looking but wrong
#      value; 0 is the value that's actually true for these rows.
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(0.0)
print(f"RESULT: TotalCharges nulls after fix: {data['TotalCharges'].isnull().sum()}")

# --- 3b. Drop the identifier -----------------------------------------
# DECISION: drop customerID.
# WHY: it's a unique key with no predictive signal — keeping it risks the
#      tree finding spurious splits on an arbitrary ID string.
data = data.drop(columns=["customerID"])

# --- 3c. Fix SeniorCitizen's type ------------------------------------
# DECISION: remap SeniorCitizen from 0/1 integers to "No"/"Yes" strings.
# WHY: it's semantically a yes/no flag like the other categoricals
#      (Partner, Dependents, ...). Leaving it numeric would let the
#      ColumnTransformer route it to the numeric branch, where it would
#      get median-imputed instead of one-hot encoded — the wrong
#      treatment for a categorical flag.
data["SeniorCitizen"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})

# --- 3d. Split features / target, identify column types --------------
X = data.drop(columns=[TARGET])
y = data[TARGET].map({"No": 0, "Yes": 1})

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()
print(
    f"RESULT: {len(categorical_columns)} categorical columns, "
    f"{len(numerical_columns)} numerical columns"
)

# --- 3e. Build the imputation + encoding pipeline ---------------------
# DECISION: median imputation for any remaining numeric gaps, most-
#           frequent imputation for categorical gaps, OneHotEncoder
#           (not LabelEncoder) for categoricals.
# WHY imputers: median is robust to outliers (e.g. MonthlyCharges spikes);
#      most-frequent is the standard default for categorical missingness.
# WHY OneHotEncoder specifically: columns like Contract, InternetService,
#      PaymentMethod have no natural order. LabelEncoder would assign them
#      arbitrary integers (0, 1, 2...), which tells a Decision Tree
#      "category 2 > category 1" — a relationship that doesn't exist and
#      would distort the splits it chooses. One-hot avoids inventing that
#      order. handle_unknown="ignore" protects against a category showing
#      up in test data that the training fold never saw.
preprocessor = ColumnTransformer(
    [
        ("num", SimpleImputer(strategy="median"), numerical_columns),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_columns,
        ),
    ],
    verbose_feature_names_out=False,
)

print(
    "RESULT: preprocessing pipeline built — fit happens inside the "
    "train/test split below, never on the full dataset, so test data "
    "never leaks into the imputer's medians/modes or the encoder's "
    "known categories."
)


# ======================================================================
# 4. WHAT I DID WITH THE PROCESSED DATA, AND WHY
# ======================================================================

section("4. TRAIN/TEST SPLIT, MODEL, TUNING")

# --- 4a. Split -----------------------------------------------------
# DECISION: 80/20 split, stratified on y.
# WHY stratify: with a 73.5/26.5 imbalance, a plain random split risks
#      a test set with a noticeably different churn ratio than training,
#      which would make the evaluation numbers unstable and not
#      comparable run to run. Stratifying keeps both sets at ~26.5% churn.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"RESULT: train={X_train.shape[0]} rows, test={X_test.shape[0]} rows")
print(
    f"Train churn rate: {y_train.mean():.3f}  |  Test churn rate: {y_test.mean():.3f}"
)

# --- 4b. Model choice ------------------------------------------------
# DECISION: DecisionTreeClassifier with class_weight="balanced".
# WHY the model: specified by the assignment, and it's interpretable —
#      I can inspect the actual splits it makes, which matters for a
#      report that has to explain the reasoning, not just report a score.
# WHY class_weight="balanced": without it, a tree trained on 73.5/26.5
#      data can reach fairly high accuracy by mostly predicting "No
#      Churn" and rarely committing to "Churn" — exactly the failure mode
#      the class balance in section 2 warned about. Balancing the class
#      weights makes misclassifying a churner cost as much as
#      misclassifying a non-churner during training.
model = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

# --- 4c. Hyperparameter tuning ---------------------------------------
# DECISION: GridSearchCV over max_depth, min_samples_leaf, criterion,
#           scored on ROC-AUC, with 5-fold stratified CV.
# WHY these hyperparameters: max_depth and min_samples_leaf are the two
#      main levers against overfitting for a single tree — an unconstrained
#      tree will happily memorize the training set. criterion (gini vs
#      entropy) is a cheap thing to search since both are fast to compute.
# WHY ROC-AUC as the search metric (not accuracy): accuracy rewards the
#      majority-class shortcut described above; ROC-AUC measures how well
#      the model ranks churners above non-churners regardless of the
#      class imbalance, which is what I actually care about here.
search = GridSearchCV(
    pipeline,
    PARAM_GRID,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1,
)
search.fit(X_train, y_train)
best_pipeline = search.best_estimator_

print(f"RESULT: best params = {search.best_params_}")
print(f"RESULT: best cross-validated ROC-AUC = {search.best_score_:.4f}")


# ======================================================================
# 5. EVALUATION — CRITERIA STATED BEFORE THE NUMBERS
# ======================================================================
#
# DECISION: report recall (churn class), F1, ROC-AUC and the confusion
#           matrix — not accuracy as the headline number.
# WHY: given the 73.5/26.5 split, a model predicting "No Churn" for
#      everyone scores ~73.5% accuracy while catching zero actual
#      churners. Since the business goal behind this assignment is
#      identifying customers likely to churn, recall on the churn class
#      (how many actual churners the model catches) and ROC-AUC (overall
#      ranking quality) are the numbers that matter — accuracy is
#      reported too, but only as context, not as the criterion.

section("5. EVALUATION")

y_pred = best_pipeline.predict(X_test)
y_probability = best_pipeline.predict_proba(X_test)[:, 1]

print("Classification report (test set):")
print(
    classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], digits=4)
)

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))

test_roc_auc = roc_auc_score(y_test, y_probability)
print(f"\nTest ROC-AUC: {test_roc_auc:.4f}")


# ======================================================================
# 6. CONFIRMING THE EVALUATION ISN'T FOOLING ME
# ======================================================================
#
# A good-looking test score alone doesn't prove the model learned
# something real or that it will generalize. Three checks:

section("6. SANITY-CHECKING THE EVALUATION")

# --- 6a. Baseline comparison -------------------------------------
# DECISION: compare against a DummyClassifier that always predicts the
#           majority class.
# WHY: this is the floor. If the tuned tree doesn't clearly beat this,
#      it hasn't learned anything useful — it's just reflecting the
#      class imbalance back.
baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
print("Baseline (always predict majority class) classification report:")
print(
    classification_report(
        y_test, baseline_pred, target_names=["No Churn", "Churn"], digits=4
    )
)
print(
    "RESULT: compare the Churn-class recall/F1 above against the tuned "
    "model's numbers in section 5 — the baseline should score 0 on "
    "Churn recall, since it never predicts that class."
)

# --- 6b. Train vs. test gap ---------------------------------------
# DECISION: compute the same metric on the training set and compare it
#           to the test-set score.
# WHY: a large train-test gap means the model memorized the training
#      data (overfitting) rather than learning a generalizable pattern,
#      even if the test score alone looks acceptable.
train_probability = best_pipeline.predict_proba(X_train)[:, 1]
train_roc_auc = roc_auc_score(y_train, train_probability)
print(f"\nTrain ROC-AUC: {train_roc_auc:.4f}  |  Test ROC-AUC: {test_roc_auc:.4f}")
print(
    f"Gap: {train_roc_auc - test_roc_auc:.4f}  "
    "(a small gap here supports that max_depth/min_samples_leaf "
    "tuning in section 4c is actually controlling overfitting)"
)

# --- 6c. Cross-validation stability --------------------------------
# DECISION: re-run 5-fold CV with the chosen hyperparameters and look at
#           the spread across folds, not just the mean.
# WHY: GridSearchCV already reported a mean CV score — but a high mean
#      with wildly different scores per fold would mean the result
#      depends heavily on which rows happened to land in which fold,
#      i.e. it isn't a stable estimate.
cv_scores = cross_val_score(
    best_pipeline,
    X_train,
    y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="roc_auc",
)
print(f"\nPer-fold ROC-AUC: {np.round(cv_scores, 4)}")
print(
    f"Mean: {cv_scores.mean():.4f}  |  Std dev: {cv_scores.std():.4f}  "
    "(a small std dev means the score isn't a fluke of one lucky split)"
)


# ======================================================================
# 7. WHAT THE TREE ACTUALLY LEARNED
# ======================================================================
#
# DECISION: report feature importances.
# WHY: this is the piece that makes the tree's decisions inspectable,
#      not just its score reportable — useful for both the report's
#      discussion section and for sanity-checking that the model is
#      keying off believable signals (e.g. tenure, contract type) rather
#      than something spurious.

section("7. FEATURE IMPORTANCES")

feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pd.Series(
    best_pipeline.named_steps["model"].feature_importances_, index=feature_names
).sort_values(ascending=False)

print("Top 10 features the tree relies on most:")
print(importances.head(10).to_string())
