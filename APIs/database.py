from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from dotenv import load_dotenv
import os
import urllib.parse
import uuid

# ✅ Load environment variables
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)

# ✅ MySQL Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://database_0x8n_user:UL2UuxdWDNKNodXT5rr5bURpALnZWE4J@dpg-cvffke5ds78s73flab60-a/database_0x8n")
# ✅ Create Engine & Session
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ Fraud Detection Table (Existing)
class FraudDetection(Base):
    __tablename__ = "fraud_detection"
    
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))  # ✅ Auto-generate unique ID
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

# ✅ Ensure Database Tables Exist
Base.metadata.create_all(bind=engine)

class FraudReport(Base):
    __tablename__ = "fraud_reporting"
    transaction_id = Column(String(50), primary_key=True, index=True, nullable=False)
    reporting_acknowledged = Column(Boolean, default=False)  # ✅ Acknowledgment flag
    failure_code = Column(Float, default=0)  # ✅ Failure code (0 = Success, 1 = Already Reported)

# ✅ Ensure the Table is Created
Base.metadata.create_all(bind=engine)