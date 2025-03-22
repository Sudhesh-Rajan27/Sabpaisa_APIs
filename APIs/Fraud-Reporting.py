from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from APIs.database import SessionLocal, FraudReport  # ✅ Ensure correct import

app = FastAPI()

# ✅ Define Fraud Report Schema
class FraudReportRequest(BaseModel):
    transaction_id: str

@app.post("/report_fraud")
async def report_fraud(fraud_report: FraudReportRequest):
    """Store a reported fraud case in the database."""
    try:
        db = SessionLocal()
        
        # ✅ Check if the transaction already exists in fraud reports
        existing_report = db.query(FraudReport).filter_by(transaction_id=fraud_report.transaction_id).first()
        
        if existing_report:
            return {
                "transaction_id": fraud_report.transaction_id,
                "reporting_acknowledged": False,
                "failure_code": 1  # 1 = Already Reported
            }
        
        # ✅ Create a new fraud report entry
        new_report = FraudReport(
            transaction_id=fraud_report.transaction_id,
            reporting_acknowledged=True,  # ✅ Set to True
            failure_code=0  # ✅ 0 = Success
        )

        # ✅ Save to database
        db.add(new_report)
        db.commit()
        db.close()

        return {
            "transaction_id": fraud_report.transaction_id,
            "reporting_acknowledged": True,
            "failure_code": 0  # 0 = Success
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
