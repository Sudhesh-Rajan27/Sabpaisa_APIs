from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import asyncio

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

class BatchRequest(BaseModel):
    transactions: list[Transaction]  # List of transactions

# URL of the Real-Time API
REAL_TIME_API_URL = "http://127.0.0.1:8000/detect_fraud"

async def process_transaction(transaction: dict):
    """Send each transaction to the Real-Time API."""
    try:
        response = await asyncio.to_thread(requests.post, REAL_TIME_API_URL, json=transaction)
        return response.json()
    except Exception as e:
        return {"transaction_id": transaction["transaction_id"], "error": str(e)}

@app.post("/detect_fraud_batch")
async def detect_fraud_batch(batch_request: BatchRequest):
    """Process multiple transactions in parallel."""
    try:
        # Convert each transaction to dict
        transactions_list = [t.dict() for t in batch_request.transactions]

        # Run all transactions in parallel
        results = await asyncio.gather(*[process_transaction(txn) for txn in transactions_list])

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
