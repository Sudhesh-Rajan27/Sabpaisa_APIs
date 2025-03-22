from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os
import uuid

# ✅ Load Database URL from Environment Variables
DATABASE_URL = os.getenv("DB_URL")

if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is not set. Make sure it's configured in Railway!")

# ✅ Create Engine & Session
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ Fraud Detection Table
class FraudDetection(Base):
    __tablename__ = "fraud_detection"
    
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id = Column(String(50), unique=True, index=True, nullable=False)
    payer_id = Column(String(50), index=True, nullable=False)
    payee_id = Column(String(50), index=True, nullable=False)
    amount = Column(Float, nullable=False)
    transaction_mode = Column(String(50), nullable=False)
    location = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    is_fraud_predicted = Column(Boolean, default=False)
    fraud_source = Column(String(20), nullable=False)
    fraud_reason = Column(String(255), nullable=True)
    fraud_score = Column(Float, nullable=False)

# ✅ Fraud Reporting Table
class FraudReport(Base):
    __tablename__ = "fraud_reporting"
    
    transaction_id = Column(String(50), primary_key=True, index=True, nullable=False)
    reporting_acknowledged = Column(Boolean, default=False)
    failure_code = Column(Float, default=0)

# ✅ Ensure Database Tables Exist
Base.metadata.create_all(bind=engine)
