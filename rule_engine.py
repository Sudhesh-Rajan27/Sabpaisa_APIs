import pandas as pd
import numpy as np
from datetime import datetime

class FraudDetectionEngine:
    def __init__(self, transaction_data):
        """
        Initialize with transaction data.
        """
        self.df = pd.DataFrame(transaction_data)
        self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
        self.df['hour_of_day'] = self.df['transaction_date'].dt.hour  # Extract transaction time
        self.suspected_transactions = set()

    def detect_multiple_card_usage(self, time_window=5):
        """
        Rule 1: Flag if multiple transactions occur within 'time_window' seconds 
        using different payment methods but from the same IP.
        """
        self.df.sort_values(by=['transaction_date'], inplace=True)
        
        flagged = set()
        for i, row in self.df.iterrows():
            for j in range(i + 1, len(self.df)):
                time_diff = abs((self.df.iloc[j]['transaction_date'] - row['transaction_date']).total_seconds())
                if (
                    time_diff <= time_window
                    and row['payee_ip_anonymous'] == self.df.iloc[j]['payee_ip_anonymous']
                    and row['transaction_payment_mode_anonymous'] != self.df.iloc[j]['transaction_payment_mode_anonymous']
                ):
                    flagged.add(row['transaction_id_anonymous'])

        self.suspected_transactions.update(flagged)
        return flagged

    def detect_multiple_payees(self, time_window=5):
        """
        Rule 2: Flag if a single device/IP is making payments to multiple payees within 'time_window' seconds.
        """
        self.df.sort_values(by=['transaction_date'], inplace=True)
        
        flagged = set()
        for i, row in self.df.iterrows():
            for j in range(i + 1, len(self.df)):
                time_diff = abs((self.df.iloc[j]['transaction_date'] - row['transaction_date']).total_seconds())
                if (
                    time_diff <= time_window
                    and row['payee_ip_anonymous'] == self.df.iloc[j]['payee_ip_anonymous']
                    and row['payee_id_anonymous'] != self.df.iloc[j]['payee_id_anonymous']
                ):
                    flagged.add(row['transaction_id_anonymous'])

        self.suspected_transactions.update(flagged)
        return flagged

    def detect_drift_anomalies(self):
        """
        Rule 3: Detect drift anomalies in:
        - Transaction amount
        - Payment method
        - Payee behavior
        - Transaction time
        """
        flagged = set()

        # --- Transaction Amount Anomaly Detection (IQR) ---
        clean_data = self.df['transaction_amount'].dropna()
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)

        flagged.update(set(self.df[self.df['transaction_amount'] > upper_bound]['transaction_id_anonymous']))

        # --- Payment Method Drift Detection ---
        recent_payment_modes = self.df['transaction_payment_mode_anonymous'].dropna().tail(10).value_counts()
        if not recent_payment_modes.empty:
            common_mode = recent_payment_modes.idxmax()
            flagged.update(set(self.df[self.df['transaction_payment_mode_anonymous'] != common_mode]['transaction_id_anonymous']))

        # --- Payee Behavior Drift Detection ---
        payee_counts = self.df['payee_id_anonymous'].value_counts()
        if not payee_counts.empty:
            high_risk_payees = payee_counts[payee_counts > payee_counts.mean() + (2 * payee_counts.std())].index
            flagged.update(set(self.df[self.df['payee_id_anonymous'].isin(high_risk_payees)]['transaction_id_anonymous']))

        # --- Transaction Time Drift Detection ---
        time_mean = self.df['hour_of_day'].mean()
        time_std = self.df['hour_of_day'].std()

        flagged.update(set(self.df[np.abs(self.df['hour_of_day'] - time_mean) > time_std]['transaction_id_anonymous']))

        self.suspected_transactions.update(flagged)
        return flagged

    def run_fraud_detection(self):
        """
        Run all fraud detection rules.
        """
        print("🚀 Running fraud detection engine...")
        fraud_1 = self.detect_multiple_card_usage()
        fraud_2 = self.detect_multiple_payees()
        fraud_3 = self.detect_drift_anomalies()

        all_frauds = fraud_1.union(fraud_2).union(fraud_3)
        
        print(f"🔴 Suspicious Transactions: {len(all_frauds)}")
        return all_frauds

# Example Transaction Data
transaction_data = [
    {'transaction_amount': 500, 'transaction_date': '2025-03-21 10:15:30', 'transaction_payment_mode_anonymous': 'Credit Card',
     'payee_ip_anonymous': '192.168.1.1', 'transaction_id_anonymous': 'TXN001', 'payee_id_anonymous': 'PAY001'},
    
    {'transaction_amount': 1200, 'transaction_date': '2025-03-21 10:15:35', 'transaction_payment_mode_anonymous': 'Debit Card',
     'payee_ip_anonymous': '192.168.1.1', 'transaction_id_anonymous': 'TXN002', 'payee_id_anonymous': 'PAY002'},
    
    {'transaction_amount': 9000, 'transaction_date': '2025-03-21 23:59:59', 'transaction_payment_mode_anonymous': 'Net Banking',
     'payee_ip_anonymous': '192.168.2.2', 'transaction_id_anonymous': 'TXN003', 'payee_id_anonymous': 'PAY003'}
]

# Run Fraud Detection Engine
engine = FraudDetectionEngine(transaction_data)
fraudulent_transactions = engine.run_fraud_detection()
print("⚠ Fraudulent Transactions Detected:", fraudulent_transactions)
