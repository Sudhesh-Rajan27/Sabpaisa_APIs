from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import httpx
import asyncio

app = FastAPI()

# ✅ Define Transaction Schema
class Transaction(BaseModel):
    transaction_id: str
    payer_id: str
    payee_id: str
    amount: float
    transaction_mode: str
    location: str
    transaction_timestamp: str

class BatchRequest(BaseModel):
    transactions: List[Transaction]  # ✅ Ensure correct List type

# ✅ URL of the Real-Time API
REAL_TIME_API_URL = "http://127.0.0.1:8000/detect_fraud"

async def process_transaction(transaction: dict):
    """Send each transaction to the Real-Time API asynchronously."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(REAL_TIME_API_URL, json=transaction)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as http_err:
            return {"transaction_id": transaction["transaction_id"], "error": f"HTTP error: {http_err.response.status_code}"}
        except Exception as e:
            return {"transaction_id": transaction["transaction_id"], "error": str(e)}

@app.post("/detect_fraud_batch")
async def detect_fraud_batch(batch_request: BatchRequest):
    """Process multiple transactions in parallel and return a dictionary of results."""
    try:
        # ✅ Convert each transaction to dictionary format
        transactions_list = [t.dict() for t in batch_request.transactions]

        # ✅ Process all transactions in parallel
        results = await asyncio.gather(*[process_transaction(txn) for txn in transactions_list])

        # ✅ Convert results into a dictionary with transaction_id as key
        fraud_results = {txn["transaction_id"]: txn for txn in results}

        return fraud_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
