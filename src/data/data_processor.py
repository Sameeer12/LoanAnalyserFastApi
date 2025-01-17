# src/data/data_processor.py
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LoanDataProcessor:
    def __init__(self):
        self.required_columns = [
            'application_id',
            'customer_id',
            'pincode',
            'applied_amount',
            'loan_type',
            'application_date',
            'income',
            'occupation',
            'status'  # 'Approved', 'Rejected', 'Inquired'
        ]

    def process_loan_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Process loan application data and organize by pincode"""
        try:
            # Validate required columns
            self._validate_columns(df)

            # Basic preprocessing
            df = self._preprocess_data(df)

            # Organize data by pincode
            pincode_data = {}
            for pincode in df['pincode'].unique():
                pincode_data[pincode] = self._process_pincode_data(df[df['pincode'] == pincode])

            return pincode_data

        except Exception as e:
            logger.error(f"Error processing loan data: {str(e)}")
            raise

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that all required columns are present"""
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic preprocessing on the data"""
        df = df.copy()

        # Convert dates
        df['application_date'] = pd.to_datetime(df['application_date'])

        # Convert numeric fields
        df['applied_amount'] = pd.to_numeric(df['applied_amount'], errors='coerce')
        df['income'] = pd.to_numeric(df['income'], errors='coerce')

        # Handle missing values
        df = df.dropna(subset=['pincode', 'status'])  # Critical columns
        df['income'] = df['income'].fillna(df['income'].median())
        df['applied_amount'] = df['applied_amount'].fillna(df['applied_amount'].median())

        return df

    def _process_pincode_data(self, df: pd.DataFrame) -> Dict:
        """Process data for a specific pincode"""
        return {
            'demographic_insights': self._extract_demographics(df),
            'loan_patterns': self._extract_loan_patterns(df),
            'performance_metrics': self._extract_performance_metrics(df)
        }

    def _extract_demographics(self, df: pd.DataFrame) -> Dict:
        """Extract demographic insights from loan applications"""
        income_quartiles = df['income'].quantile([0.25, 0.5, 0.75]).to_dict()

        return {
            'total_applicants': len(df),
            'unique_customers': df['customer_id'].nunique(),
            'income_distribution': {
                'low': len(df[df['income'] <= income_quartiles[0.25]]),
                'medium': len(df[(df['income'] > income_quartiles[0.25]) &
                                 (df['income'] <= income_quartiles[0.75])]),
                'high': len(df[df['income'] > income_quartiles[0.75]])
            },
            'occupation_distribution': df['occupation'].value_counts().to_dict(),
            'avg_income': df['income'].mean(),
            'median_income': df['income'].median()
        }

    def _extract_loan_patterns(self, df: pd.DataFrame) -> Dict:
        """Extract loan application patterns"""
        return {
            'loan_type_distribution': df['loan_type'].value_counts().to_dict(),
            'amount_metrics': {
                'mean': df['applied_amount'].mean(),
                'median': df['applied_amount'].median(),
                'min': df['applied_amount'].min(),
                'max': df['applied_amount'].max(),
                'quartiles': df['applied_amount'].quantile([0.25, 0.5, 0.75]).to_dict()
            },
            'loan_type_amount_patterns': {
                loan_type: {
                    'avg_amount': group['applied_amount'].mean(),
                    'count': len(group)
                }
                for loan_type, group in df.groupby('loan_type')
            }
        }

    def _extract_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Extract performance metrics from loan applications"""
        total_applications = len(df)

        # Calculate status ratios
        status_counts = df['status'].value_counts()
        status_ratios = {
            status: count / total_applications
            for status, count in status_counts.items()
        }

        # Calculate loan type success rates
        loan_type_metrics = {}
        for loan_type in df['loan_type'].unique():
            loan_type_data = df[df['loan_type'] == loan_type]
            loan_type_metrics[loan_type] = {
                'approval_rate': len(loan_type_data[loan_type_data['status'] == 'Approved']) / len(loan_type_data),
                'rejection_rate': len(loan_type_data[loan_type_data['status'] == 'Rejected']) / len(loan_type_data),
                'inquiry_rate': len(loan_type_data[loan_type_data['status'] == 'Inquired']) / len(loan_type_data)
            }

        # Extract temporal patterns
        df['month'] = df['application_date'].dt.to_period('M')
        monthly_patterns = df.groupby('month').agg({
            'application_id': 'count',
            'status': lambda x: (x == 'Approved').mean()
        }).to_dict()

        return {
            'overall_metrics': status_ratios,
            'loan_type_metrics': loan_type_metrics,
            'temporal_patterns': monthly_patterns,
            'amount_success_correlation': self._calculate_amount_success_correlation(df)
        }

    def _calculate_amount_success_correlation(self, df: pd.DataFrame) -> Dict:
        """Calculate correlation between loan amount and success rate"""
        df['amount_quartile'] = pd.qcut(df['applied_amount'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        success_by_amount = df.groupby('amount_quartile').agg({
            'status': lambda x: (x == 'Approved').mean()
        }).to_dict()['status']

        return {
            'amount_quartile_success_rates': success_by_amount
        }