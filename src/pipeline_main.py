"""Train and evaluate a decision tree for telecom customer churn prediction."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
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


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset, clean it, and split it into features and target."""
    data = pd.read_csv(path)

    # TotalCharges contains blank strings for customers with zero tenure.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # SeniorCitizen is stored as 0/1 but is a yes/no flag like the other categoricals.
    data["SeniorCitizen"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})

    X = data.drop(columns=["customerID", TARGET])
    y = data[TARGET].map({"No": 0, "Yes": 1})
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build the preprocessing and decision-tree pipeline."""
    categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_columns = X.select_dtypes(include="number").columns.tolist()

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

    # About 26% of customers churn, so weight classes to keep the tree from
    # favouring the majority "No Churn" class.
    model = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def tune(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Select tree hyperparameters with stratified cross-validation on the training set."""
    search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print("Best parameters:", search.best_params_)
    print(f"Cross-validated ROC-AUC: {search.best_score_:.4f}\n")
    return search.best_estimator_


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print test-set metrics for the fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["No Churn", "Churn"], digits=4
        )
    )

    print("Confusion Matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nROC-AUC: {roc_auc_score(y_test, y_probability):.4f}")


def print_feature_importances(pipeline: Pipeline, top_n: int = 10) -> None:
    """Print the features the tree relies on most."""
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(
        pipeline.named_steps["model"].feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    print(f"\nTop {top_n} feature importances:")
    print(importances.head(top_n).to_string())


def main() -> None:
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = tune(build_pipeline(X), X_train, y_train)
    evaluate(pipeline, X_test, y_test)
    print_feature_importances(pipeline)


if __name__ == "__main__":
    main()
