from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from dotenv import load_dotenv
import os
import urllib.parse

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)

# MySQL Database Configuration
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# Create Engine & Session
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Fraud Detection Table
class FraudDetection(Base):
    __tablename__ = "fraud_detection"
    id = Column(String(50), primary_key=True)
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

# Create Tables in MySQL
Base.metadata.create_all(bind=engine)
