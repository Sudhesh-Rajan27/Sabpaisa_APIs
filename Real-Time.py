from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import datetime
from database import SessionLocal, FraudDetection
from rule_engine import FraudDetectionEngine  # Import your fraud detection engine
import tensorflow as tf
import numpy as np

app = FastAPI()

# Define Transaction Schema
class Transaction(BaseModel):
    transaction_id: str
    payer_id: str
    payee_id: str
    amount: float
    transaction_mode: str
    location: str
    timestamp: str

# Load AI model globally (to avoid reloading on every request)
model = tf.keras.models.load_model("gnn_fraud_detection.h5")

async def run_rule_engine(transaction_data):
    """Runs the rule-based fraud detection asynchronously."""
    rule_engine = FraudDetectionEngine(transaction_data)
    fraud_transactions = rule_engine.run_fraud_detection()
    return fraud_transactions

async def run_ai_model(input_data):
    """Runs the AI-based fraud detection asynchronously."""
    prediction = model.predict(input_data)
    fraud_score = float(prediction[0][0])
    is_fraud = fraud_score > 0.5
    return is_fraud, fraud_score

@app.post("/detect_fraud")
async def detect_fraud(transaction: Transaction):
    try:
        # Convert transaction data to the format required by the rule engine
        transaction_data = [{
            "transaction_amount": transaction.amount,
            "transaction_date": transaction.timestamp,
            "transaction_payment_mode_anonymous": transaction.transaction_mode,
            "payee_ip_anonymous": transaction.location,  
            "transaction_id_anonymous": transaction.transaction_id,
            "payee_id_anonymous": transaction.payee_id
        }]

        # Prepare AI model input
        input_data = np.array([[
            transaction.amount,             # Transaction amount
            float(transaction.timestamp.split("-")[0]),  # Extract year from timestamp
            hash(transaction.transaction_mode) % 100,    # Convert mode to a number
            hash(transaction.payer_id) % 100,            # Convert payer_id to a number
            hash(transaction.payee_id) % 100             # Convert payee_id to a number
        ]])

        # Run Rule-Based Detection & AI Model in Parallel
        rule_engine_task = run_rule_engine(transaction_data)
        ai_model_task = run_ai_model(input_data)
        rule_fraud_transactions, (ai_is_fraud, fraud_score) = await asyncio.gather(rule_engine_task, ai_model_task)

        # Determine final fraud decision
        if transaction.transaction_id in rule_fraud_transactions:
            fraud_source = "Rule"
            fraud_reason = "Matched fraud pattern"
            is_fraud = True
            fraud_score = 1.0  # Rule-based fraud is deterministic
        else:
            fraud_source = "AI Model"
            fraud_reason = "AI model detected suspicious transaction" if ai_is_fraud else "Transaction appears normal"
            is_fraud = ai_is_fraud

        # Store result in the MySQL database
        db = SessionLocal()
        fraud_record = FraudDetection(
            id=transaction.transaction_id,
            transaction_id=transaction.transaction_id,
            payer_id=transaction.payer_id,
            payee_id=transaction.payee_id,
            amount=transaction.amount,
            transaction_mode=transaction.transaction_mode,
            location=transaction.location,
            timestamp=datetime.datetime.fromisoformat(transaction.timestamp),
            is_fraud_predicted=is_fraud,
            fraud_source=fraud_source,
            fraud_reason=fraud_reason,
            fraud_score=fraud_score
        )
        db.add(fraud_record)
        db.commit()
        db.close()

        # Return API Response
        return {
            "transaction_id": transaction.transaction_id,
            "is_fraud": is_fraud,
            "fraud_source": fraud_source,
            "fraud_reason": fraud_reason,
            "fraud_score": fraud_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
