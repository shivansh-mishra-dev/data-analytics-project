import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1. Data Loading and Initial Understanding
# ==========================================

# Load telecom customer churn dataset into a pandas DataFrame
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Inspect dataset dimensions and preview columns
print("Dataset Shape:", df.shape)
pd.set_option("display.max_columns", None)
print(df.head())
print(df.info())

# Drop customerID column as it is an identifier not required for modeling
df = df.drop(columns=["customerID"])

# Identify unique values across categorical features
numerical_features_list = ["tenure", "MonthlyCharges", "TotalCharges"]
for col in df.columns:
    if col not in numerical_features_list:
        print(f"{col}: {df[col].unique()}")
        print("-" * 50)

# Replace empty space strings in TotalCharges with '0.0' and convert to float
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)


# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================


# Function to plot distribution histogram with KDE, mean, and median lines
def plot_histogram(data, column_name):
    plt.figure(figsize=(5, 3))
    sns.histplot(data[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")
    col_mean = data[column_name].mean()
    col_median = data[column_name].median()
    plt.axvline(col_mean, color="red", linestyle="--", label="Mean")
    plt.axvline(col_median, color="green", linestyle="-", label="Median")
    plt.legend()
    plt.show()


for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    plot_histogram(df, col)


# Function to plot box plot for detecting outliers in numerical features
def plot_boxplot(data, column_name):
    plt.figure(figsize=(5, 3))
    sns.boxplot(y=data[column_name])
    plt.title(f"Box Plot of {column_name}")
    plt.ylabel(column_name)
    plt.show()


for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    plot_boxplot(df, col)

# Correlation heatmap for numerical columns
plt.figure(figsize=(8, 4))
sns.heatmap(
    df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)
plt.title("Correlation Heatmap")
plt.show()

# Count plot for categorical columns (including SeniorCitizen)
object_cols_eda = df.select_dtypes(include="object").columns.to_list()
object_cols_eda = ["SeniorCitizen"] + object_cols_eda

for col in object_cols_eda:
    plt.figure(figsize=(5, 3))
    sns.countplot(x=df[col])
    plt.title(f"Count Plot of {col}")
    plt.show()


# ==========================================
# 3. Data Preprocessing
# ==========================================

# Encode target variable: Yes -> 1, No -> 0
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

# Identify categorical columns to encode
object_cols = df.select_dtypes(include="object").columns.to_list()
object_cols = ["SeniorCitizen"] + object_cols

# Initialize LabelEncoders, fit-transform columns, and save encoders to a dictionary
encoders = {}
for column in object_cols:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

# Save encoders dictionary to a pickle file for inference
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)


# ==========================================
# 4. Training and Test Data Split
# ==========================================

# Split into features (X) and target (y)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================
# 5. Model Training & Evaluation (Decision Tree)
# ==========================================

# Initialize Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Perform 5-fold cross-validation on the training data
cv_scores = cross_val_score(dt_classifier, X_train, y_train, cv=5, scoring="accuracy")
print("Training Decision Tree with default parameters")
print(f"Decision Tree cross-validation accuracy: {np.mean(cv_scores):.2f}")
print("CV Scores:", cv_scores)

# Train the model on the full training dataset
dt_classifier.fit(X_train, y_train)

# Evaluate model performance on unseen test data
y_test_pred = dt_classifier.predict(X_test)
print("\n--- Model Evaluation on Test Data ---")
print("Accuracy Score:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Save the trained Decision Tree model along with feature names
model_data = {"model": dt_classifier, "features_names": X.columns.tolist()}
with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)


# ==========================================
# 6. Load Saved Model and Build Predictive System
# ==========================================

# Load the saved model and feature list from pickle file
with open("customer_churn_model.pkl", "rb") as f:
    loaded_model_data = pickle.load(f)

loaded_model = loaded_model_data["model"]
feature_names = loaded_model_data["features_names"]

# Load the saved label encoders
with open("encoders.pkl", "rb") as f:
    loaded_encoders = pickle.load(f)

# Sample unseen customer input data
input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}

# Convert input dictionary into DataFrame
input_data_df = pd.DataFrame([input_data])

# Encode categorical variables using loaded encoders
for column, encoder in loaded_encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

# Ensure columns align with training feature order (prevents silent misalignment)
input_data_df = input_data_df[feature_names]

# Make prediction and compute prediction probabilities
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

print("\n--- Prediction Result ---")
print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediction Probability: {pred_prob}")
