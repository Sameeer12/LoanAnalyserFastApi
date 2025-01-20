# src/api/models.py
from typing import List, Dict, Optional

from pydantic import BaseModel


class PincodeRequest(BaseModel):
    pincode: str
    include_historical: bool = False


class LoanApplication(BaseModel):
    application_id: str
    customer_id: str
    pincode: str
    applied_amount: float
    loan_type: str
    loan_start_date: str
    income: float
    occupation: str
    status: str


class AnalysisRequest(BaseModel):
    loan_data: List[LoanApplication]
    pincodes: Optional[List[str]] = None


class StrategyResponse(BaseModel):
    pincode: str
    market_analysis: Dict
    strategy_recommendations: Dict
