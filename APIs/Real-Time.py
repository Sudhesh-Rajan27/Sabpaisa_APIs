from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import datetime
import numpy as np
import joblib
from APIs.database import SessionLocal, FraudDetection
from Models.rule_engine import FraudDetectionEngine
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__),"../Models/gnn_fraud_model.pt")
app = FastAPI()

# ✅ Define GNNFraudModel (Same as in Training Script)
class GNNFraudModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNFraudModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.batch_norm(x) if x.shape[0] > 1 else x
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)  # Binary classification

# ✅ Load the trained GNN model
input_dim = 5  # Ensure this matches the model's input feature count
model = GNNFraudModel(input_dim, hidden_dim=8, output_dim=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# ✅ Load the StandardScaler for input normalization
SCALER_PATH = os.path.join(os.path.dirname(__file__),"../Models/scaler.pkl")
scaler = joblib.load(SCALER_PATH)

# ✅ Define the Transaction Schema
class Transaction(BaseModel):
    transaction_id: str
    payer_id: str
    payee_id: str
    amount: float
    transaction_mode: str
    location: str
    transaction_timestamp: str  # ✅ Matches database column

async def run_rule_engine(transaction_data):
    """Runs the Rule-Based Fraud Detection Engine."""
    rule_engine = FraudDetectionEngine(transaction_data)
    fraud_transactions = rule_engine.run_fraud_detection()
    return fraud_transactions

async def run_ai_model(transaction_features):
    """Runs the new GNN-based AI Fraud Detection Model."""
    # ✅ Standardize input data using the saved scaler
    standardized_features = scaler.transform([transaction_features])

    # ✅ Convert to PyTorch tensor
    input_tensor = torch.tensor(standardized_features, dtype=torch.float).unsqueeze(0)

    # ✅ Ensure edge index follows PyG's expected shape
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop

    # ✅ Create PyG Data object
    transaction_graph = torch_geometric.data.Data(x=input_tensor, edge_index=edge_index)

    # ✅ Get fraud prediction
    with torch.no_grad():
        fraud_score = model(transaction_graph).item()

    return fraud_score

@app.post("/detect_fraud")
async def detect_fraud(transaction: Transaction):
    """Detects fraud using Rule-Based and AI Fraud Detection."""
    try:
        # ✅ Debugging
        print(f"🔹 Received Transaction Data: {transaction.dict()}")

        # ✅ Extract year from timestamp
        transaction_year = float(transaction.transaction_timestamp.split("-")[0])

        # ✅ Convert transaction data into dictionary format
        transaction_data = [{
            "transaction_amount": transaction.amount,
            "transaction_timestamp": transaction.transaction_timestamp,
            "transaction_payment_mode": transaction.transaction_mode,
            "payee_ip": transaction.location,
            "transaction_id": transaction.transaction_id,
            "payee_id": transaction.payee_id
        }]

        # ✅ Prepare AI Model Input Data
        transaction_features = [
            transaction.amount,
            10000,  # Placeholder for account balance (if not provided)
            hash(transaction.transaction_mode) % 10,
            hash(transaction.payer_id) % 50,
            hash(transaction.payee_id) % 50
        ]

        print(f"🔹 Final AI Model Input Data: {transaction_features}")

        # ✅ Run Rule-Based Detection & AI Model in Parallel
        rule_engine_task = run_rule_engine(transaction_data)
        ai_model_task = run_ai_model(transaction_features)
        rule_fraud_transactions, fraud_score = await asyncio.gather(rule_engine_task, ai_model_task)

        # ✅ Determine Initial Fraud Source Based on AI Model
        is_fraud = fraud_score > 0.5
        fraud_source = "AI Model" if is_fraud else "Rule Engine"
        fraud_reason = "AI Model flagged suspicious activity" if is_fraud else "No suspicious patterns detected by AI."

        # ✅ If Rule Engine Flags Fraud, Override AI Decision
        if transaction.transaction_id in rule_fraud_transactions:
            is_fraud = True  # Ensure fraud flag remains True
            fraud_source = "Rule Engine"
            fraud_reason = "Rule Engine identified fraudulent behavior."

        # ✅ Final Decision Output
        print(f"🔍 Fraud Decision: {is_fraud} | Source: {fraud_source} | Reason: {fraud_reason}")

        # ✅ Store Results in Database
        db = SessionLocal()
        fraud_record = FraudDetection(
            transaction_id=transaction.transaction_id,
            payer_id=transaction.payer_id,
            payee_id=transaction.payee_id,
            amount=transaction.amount,
            transaction_mode=transaction.transaction_mode,
            location=transaction.location,
            timestamp=datetime.datetime.fromisoformat(transaction.transaction_timestamp),
            is_fraud_predicted=is_fraud,
            fraud_source=fraud_source,
            fraud_reason=fraud_reason,
            fraud_score=fraud_score
        )
        db.add(fraud_record)
        db.commit()
        db.close()

        # ✅ Return API Response
        return {
            "transaction_id": transaction.transaction_id,
            "is_fraud": is_fraud,
            "fraud_source": fraud_source,
            "fraud_reason": fraud_reason,
            "fraud_score": round(fraud_score, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
