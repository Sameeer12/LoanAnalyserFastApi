import logging
from pathlib import Path

import pandas as pd
import yaml
from fastapi import HTTPException, APIRouter
from starlette.responses import JSONResponse

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


@router.get("/pincodes")
async def get_pincodes():
    """
    Fetch all pincodes with their metrics from loan applications data.
    Calculates metrics based on loan applications, status, payments, and NPA data.
    """
    try:
        # Load loan applications data
        df = pd.read_csv("data/loan_applications.csv")

        # Group by pincode and calculate metrics
        pincode_data = []

        for pincode in df['pincode'].unique():
            pincode_df = df[df['pincode'] == pincode]

            # Calculate metrics for each pincode
            total_applications = len(pincode_df)
            approved_applications = len(pincode_df[pincode_df['status'] == 'approved'])

            metrics = {
                'total_applications': total_applications,
                'approval_rate': round(
                    (approved_applications / total_applications * 100) if total_applications > 0 else 0,
                    2
                ),
                'avg_applied_amount': round(pincode_df['applied_amount'].mean(), 2),
                'avg_income': round(pincode_df['income'].mean(), 2),
                'avg_interest_rate': round(pincode_df['interest_rate'].mean(), 2),
                'avg_tenure': round(pincode_df['tenure_months'].mean(), 2),
                'payment_performance': {
                    'successful_ratio': round(
                        pincode_df['successful_payments'].mean() / pincode_df['total_payments'].mean() * 100
                        if pincode_df['total_payments'].mean() > 0 else 0,
                        2
                    ),
                    'delay_ratio': round(
                        pincode_df['delayed_payments'].mean() / pincode_df['total_payments'].mean() * 100
                        if pincode_df['total_payments'].mean() > 0 else 0,
                        2
                    )
                },
                'npa_rate': round(
                    len(pincode_df[pincode_df['npa_flag'] == 1]) / total_applications * 100
                    if total_applications > 0 else 0,
                    2
                ),
                'loan_type_distribution': pincode_df['loan_type'].value_counts().to_dict()
            }

            pincode_data.append({
                'pincode': str(pincode),
                'count': total_applications,
                'metrics': metrics,
                'occupation_distribution': pincode_df['occupation'].value_counts().to_dict()
            })

        # Sort by count in descending order
        pincode_data = sorted(pincode_data, key=lambda x: x['count'], reverse=True)

        return JSONResponse(content={
            'status': 'success',
            'data': pincode_data,
            'total_pincodes': len(pincode_data)
        })

    except FileNotFoundError:
        logger.error("Loan applications CSV file not found")
        raise HTTPException(
            status_code=500,
            detail="Data source file not found"
        )
    except Exception as e:
        logger.error(f"Error processing pincode data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing pincode data"
        )

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
