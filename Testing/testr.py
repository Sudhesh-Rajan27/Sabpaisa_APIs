import pandas as pd
from Models.rule_engine import FraudDetectionEngine

# ✅ Load test transactions
df = pd.read_csv("transaction_testr.csv")

# ✅ Convert timestamps to datetime
df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], unit="s")

# ✅ Convert DataFrame to dictionary format for rule engine
test_transactions = df.to_dict(orient="records")

# ✅ Initialize & Run the Rule Engine
rule_engine = FraudDetectionEngine(test_transactions)
fraudulent_transactions = rule_engine.run_fraud_detection()

# ✅ Print Detected Fraudulent Transactions
print("🚀 Detected Fraudulent Transactions by Rule Engine:")
for txn in fraudulent_transactions:
    print(f"🔴 {txn}")
