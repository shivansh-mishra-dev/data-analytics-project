import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data = pd.read_csv("./dataset/data.csv")


# --------------------------------------------------
# 2. Clean TotalCharges
# --------------------------------------------------

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")


# --------------------------------------------------
# 3. Remove customer ID
# --------------------------------------------------

data = data.drop(columns=["customerID"])


# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

X = data.drop(columns=["Churn"])

y = data["Churn"].map({"No": 0, "Yes": 1})


# --------------------------------------------------
# 5. Identify column types
# --------------------------------------------------

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()


# --------------------------------------------------
# 6. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ]
)


# --------------------------------------------------
# 7. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# --------------------------------------------------
# 8. Create Decision Tree
# --------------------------------------------------

model = DecisionTreeClassifier(max_depth=5, random_state=42)


# --------------------------------------------------
# 9. Create complete pipeline
# --------------------------------------------------

pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])


# --------------------------------------------------
# 10. Train
# --------------------------------------------------

pipeline.fit(X_train, y_train)


# --------------------------------------------------
# 11. Predict
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# 12. Evaluate
# --------------------------------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))


# --------------------------------------------------
# 13. ROC-AUC
# --------------------------------------------------

y_probability = pipeline.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_probability))
