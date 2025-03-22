from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ✅ Load dataset
df = pd.read_csv("transactions_train.csv")

# ✅ Ensure 'is_fraud' column exists
if "is_fraud" not in df.columns:
    raise ValueError("❌ ERROR: 'is_fraud' column is missing in the dataset!")

# ✅ Check initial fraud percentage
fraud_percentage = (df["is_fraud"].sum() / len(df)) * 100
print(f"🔹 Initial Fraud Percentage: {fraud_percentage:.2f}%")

# ✅ Convert 'transaction_date' to numeric timestamp
if "transaction_date" in df.columns:
    df["transaction_timestamp"] = pd.to_datetime(df["transaction_date"], errors="coerce").astype(int) // 10**9
    df.drop(columns=["transaction_date"], inplace=True)  # Remove original date column

# ✅ Convert 'transaction_mode' (W = Website, M = Mobile) to numeric
if "transaction_mode" in df.columns:
    mode_mapping = {"W": 0, "Website": 0, "M": 1, "Mobile": 1}
    df["transaction_mode"] = df["transaction_mode"].map(mode_mapping)

# ✅ Encode categorical variables (if any)
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))  # Convert to string before encoding

# ✅ Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# ✅ Separate features & labels
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

# ✅ Increase fraud cases dynamically
if fraud_percentage < 5:
    smote = SMOTE(sampling_strategy=0.3, random_state=42)  # Make fraud cases 30%
elif fraud_percentage < 10:
    smote = SMOTE(sampling_strategy=0.3, random_state=42)  # Make fraud cases 30%
else:
    smote = SMOTE(sampling_strategy=0.3, random_state=42)  # Make fraud cases 30%

X_resampled, y_resampled = smote.fit_resample(X, y)

# ✅ Save the new balanced dataset
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=["is_fraud"])], axis=1)
df_resampled.to_csv("transactions_train_balanced.csv", index=False)

# ✅ Print final fraud distribution
new_fraud_percentage = (df_resampled["is_fraud"].sum() / len(df_resampled)) * 100
print(f"✅ New dataset saved! Final Fraud Percentage: {new_fraud_percentage:.2f}%")
