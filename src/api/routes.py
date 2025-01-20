import logging
from pathlib import Path

import pandas as pd
import yaml
from fastapi import HTTPException, APIRouter

from src.api.models import PincodeRequest, StrategyResponse
from src.services.loan_strategy_app import LoanStrategyApp

# Load config
config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
# app = FastAPI(
#     title="Loan Strategy Analyzer API",
#     description="API for analyzing loan data and generating marketing strategies",
#     version="1.0.0"
# )
router = APIRouter()

# Configure CORS


# Initialize loan strategy app
strategy_app = LoanStrategyApp()

logger = logging.getLogger(__name__)


@router.post("/analyze/pincode", response_model=StrategyResponse)
async def analyze_pincode(request: PincodeRequest):
    """Analyze a single pincode"""
    try:
        # Load loan data (in production, this would come from a database)
        loan_data = pd.read_csv("data/loan_applications.csv")

        result = await strategy_app.analyze_pincode(loan_data, request.pincode)
        return result
    except Exception as e:
        logger.error(f"Error analyzing pincode: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/analyze/batch", response_model=Dict[str, StrategyResponse])
# async def analyze_multiple_pincodes(request: AnalysisRequest):
#     """Analyze multiple pincodes"""
#     try:
#         # Convert loan data to DataFrame
#         loan_data = pd.DataFrame([app.dict() for app in request.loan_data])
#
#         results = await strategy_app.batch_analyze_pincodes(
#             loan_data,
#             request.pincodes
#         )
#         return results
#     except Exception as e:
#         logger.error(f"Error in batch analysis: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#

@router.get("/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "version": config['app']['version']
    }
