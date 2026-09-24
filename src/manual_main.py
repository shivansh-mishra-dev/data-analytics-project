"""Decision tree for telecom customer churn, with preprocessing and tuning done by hand.

This mirrors main.py but replaces sklearn's Pipeline, ColumnTransformer,
SimpleImputer, OneHotEncoder and GridSearchCV with explicit code, so each step
is visible.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).parent / "dataset" / "telco_customer_churn.csv"
TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_samples_leaf": [1, 10, 25, 50],
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


class Preprocessor:
    """Impute missing values and one-hot encode categoricals.

    All statistics (medians, modes, category lists) are learned in fit() from
    the training data only, then reused in transform(), so no information from
    validation or test rows leaks into training.
    """

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
        self.numerical_columns = X.select_dtypes(include="number").columns.tolist()

        self.medians = X[self.numerical_columns].median()
        self.modes = X[self.categorical_columns].mode().iloc[0]
        self.categories = {
            column: sorted(X[column].dropna().unique()) for column in self.categorical_columns
        }

        self.feature_names = self.numerical_columns + [
            f"{column}_{category}"
            for column, categories in self.categories.items()
            for category in categories
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = X[self.numerical_columns].fillna(self.medians)
        categorical = X[self.categorical_columns].fillna(self.modes)

        # One column per category seen in training; a category that only
        # appears in new data gets all zeros, like handle_unknown="ignore".
        encoded = {
            f"{column}_{category}": (categorical[column] == category).astype(float)
            for column, categories in self.categories.items()
            for category in categories
        }

        return pd.concat([numeric, pd.DataFrame(encoded, index=X.index)], axis=1)[
            self.feature_names
        ]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def build_model(**params) -> DecisionTreeClassifier:
    # About 26% of customers churn, so weight classes to keep the tree from
    # favouring the majority "No Churn" class.
    return DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE, **params)


def cross_validate(params: dict, X: pd.DataFrame, y: pd.Series) -> float:
    """Return the mean ROC-AUC of one parameter combination over stratified folds."""
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_index, validation_index in folds.split(X, y):
        X_fold_train, X_validation = X.iloc[train_index], X.iloc[validation_index]
        y_fold_train, y_validation = y.iloc[train_index], y.iloc[validation_index]

        # Refit preprocessing inside each fold so the validation fold stays unseen.
        preprocessor = Preprocessor().fit(X_fold_train)
        model = build_model(**params).fit(preprocessor.transform(X_fold_train), y_fold_train)

        probability = model.predict_proba(preprocessor.transform(X_validation))[:, 1]
        scores.append(roc_auc_score(y_validation, probability))

    return float(np.mean(scores))


def tune(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Grid-search tree hyperparameters and return the best combination."""
    best_params, best_score = None, -np.inf

    for values in product(*PARAM_GRID.values()):
        params = dict(zip(PARAM_GRID.keys(), values))
        score = cross_validate(params, X_train, y_train)
        if score > best_score:
            best_params, best_score = params, score

    print("Best parameters:", best_params)
    print(f"Cross-validated ROC-AUC: {best_score:.4f}\n")
    return best_params


def evaluate(model: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print test-set metrics for the fitted model."""
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], digits=4))

    print("Confusion Matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nROC-AUC: {roc_auc_score(y_test, y_probability):.4f}")


def print_feature_importances(
    model: DecisionTreeClassifier, feature_names: list[str], top_n: int = 10
) -> None:
    """Print the features the tree relies on most."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )

    print(f"\nTop {top_n} feature importances:")
    print(importances.head(top_n).to_string())


def main() -> None:
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    best_params = tune(X_train, y_train)

    # Refit on the full training set with the chosen parameters.
    preprocessor = Preprocessor().fit(X_train)
    model = build_model(**best_params).fit(preprocessor.transform(X_train), y_train)

    evaluate(model, preprocessor.transform(X_test), y_test)
    print_feature_importances(model, preprocessor.feature_names)


if __name__ == "__main__":
    main()
