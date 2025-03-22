import pandas as pd
import numpy as np
from datetime import datetime

class FraudDetectionEngine:
    def __init__(self, transaction_data):
        self.df = pd.DataFrame(transaction_data)
        print(f"🔍 Debugging Rule Engine Input: {transaction_data}")  # Debug input data
        # 🔹 Fix: Rename `transaction_timestamp` to match dataset
        if 'transaction_timestamp' in self.df.columns:
            self.df['transaction_timestamp'] = pd.to_datetime(self.df['transaction_timestamp'])
            self.df['hour_of_day'] = self.df['transaction_timestamp'].dt.hour
        else:
            raise ValueError("❌ ERROR: 'transaction_timestamp' column is missing in the input data!")

        self.suspected_transactions = set()

    def detect_multiple_card_usage(self, time_window=5):
        self.df = self.df.sort_values(by=['transaction_timestamp'])
        self.df['prev_transaction_timestamp'] = self.df.groupby('payee_ip')['transaction_timestamp'].shift(1)
        self.df['prev_transaction_mode'] = self.df.groupby('payee_ip')['transaction_payment_mode'].shift(1)
        self.df['time_diff'] = (self.df['transaction_timestamp'] - self.df['prev_transaction_timestamp']).dt.total_seconds()
        
        flagged = set(self.df[
            (self.df['time_diff'] <= time_window) &
            (self.df['transaction_payment_mode'] != self.df['prev_transaction_mode'])
        ]['transaction_id'])
        
        self.suspected_transactions.update(flagged)
        return flagged

    def detect_multiple_payees(self, time_window=5):
        self.df = self.df.sort_values(by=['transaction_timestamp'])
        self.df['prev_payee_id'] = self.df.groupby('payee_ip')['payee_id'].shift(1)
        self.df['time_diff'] = (self.df['transaction_timestamp'] - self.df.groupby('payee_ip')['transaction_timestamp'].shift(1)).dt.total_seconds()
        
        flagged = set(self.df[
            (self.df['time_diff'] <= time_window) &
            (self.df['payee_id'] != self.df['prev_payee_id'])
        ]['transaction_id'])
        
        self.suspected_transactions.update(flagged)
        return flagged

    def detect_drift_anomalies(self):
        flagged = set()
        Q1, Q3 = self.df['transaction_amount'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - (1.5 * IQR), Q3 + (1.5 * IQR)
        
        flagged.update(set(self.df[self.df['transaction_amount'] > upper_bound]['transaction_id']))

        try:
            common_mode = self.df['transaction_payment_mode'].mode().iat[0]
            flagged.update(set(self.df[self.df['transaction_payment_mode'] != common_mode]['transaction_id']))
        except IndexError:
            pass

        time_mean, time_std = self.df['hour_of_day'].mean(), self.df['hour_of_day'].std()
        flagged.update(set(self.df[np.abs(self.df['hour_of_day'] - time_mean) > time_std]['transaction_id']))
        
        self.suspected_transactions.update(flagged)
        return flagged

    def run_fraud_detection(self):
        fraud_1 = self.detect_multiple_card_usage()
        fraud_2 = self.detect_multiple_payees()
        fraud_3 = self.detect_drift_anomalies()
        return fraud_1.union(fraud_2).union(fraud_3)
