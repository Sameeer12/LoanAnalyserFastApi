import logging
from dataclasses import dataclass
from typing import Dict

import pandas as pd

# Configure logging with more detailed format
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class DataFrameStats:
    """Helper class to store DataFrame statistics"""
    total_rows: int
    unique_values: int
    min_value: float
    max_value: float
    has_nulls: bool


class LoanDataProcessor:
    def __init__(self, min_records_for_analysis: int = 2, enable_small_data_processing: bool = True):
        self.min_records_for_analysis = min_records_for_analysis
        self.enable_small_data_processing = enable_small_data_processing

        # Required columns with their expected types
        self.required_columns = {
            'application_id': str,
            'customer_id': str,
            'pincode': str,
            'applied_amount': float,
            'loan_type': str,
            'loan_start_date': 'datetime64[ns]',
            'income': float,
            'occupation': str,
            'status': str
        }

    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely perform division, handling zero denominator case"""
        try:
            if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
                return default
            return float(numerator) / float(denominator)
        except Exception:
            return default

    def process_loan_data(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Process loan application data and organize by pincode"""
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided")
                return {}

            df = df.copy()
            self._validate_and_clean_data(df)

            pincode_data = {}
            for pincode in df['pincode'].unique():
                pincode_mask = df['pincode'] == pincode
                pincode_df = df[pincode_mask].copy()

                record_count = len(pincode_df)
                logger.info(f"Processing pincode {pincode} with {record_count} records")

                try:
                    if record_count < self.min_records_for_analysis:
                        if not self.enable_small_data_processing:
                            logger.warning(f"Insufficient records for pincode {pincode}. Skipping analysis.")
                            continue
                        logger.info(f"Using small dataset processing for pincode {pincode}")
                        pincode_data[str(pincode)] = self._process_small_dataset(pincode_df)
                    else:
                        logger.info(f"Using full analysis for pincode {pincode}")
                        pincode_data[str(pincode)] = self._process_full_dataset(pincode_df)
                except Exception as e:
                    logger.error(f"Error processing pincode {pincode}: {str(e)}")
                    continue

            return pincode_data
        except Exception as e:
            logger.error(f"Error processing loan data: {str(e)}")
            raise

    def _validate_and_clean_data(self, df: pd.DataFrame) -> None:
        """Validate and clean the input data"""
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert columns to correct types and handle missing values
        for col, dtype in self.required_columns.items():
            try:
                if dtype == 'datetime64[ns]':
                    df[col] = self._convert_to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.error(f"Error converting column {col} to {dtype}: {str(e)}")
                raise

        # Handle missing numeric values
        numeric_columns = ['applied_amount', 'income']
        for col in numeric_columns:
            if df[col].isna().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logger.info(f"Filled {df[col].isna().sum()} missing values in {col}")

    def _process_full_dataset(self, df: pd.DataFrame) -> Dict:
        """Process dataset with comprehensive metrics"""
        try:
            # Calculate basic metrics first
            total_records = len(df)
            if total_records == 0:
                return self._get_empty_full_dataset_metrics()

            # Calculate status distribution safely
            status_counts = df['status'].value_counts()
            total_applications = len(df)
            status_distribution = {
                status: self.safe_divide(count, total_applications)
                for status, count in status_counts.items()
            }

            # Calculate loan type metrics
            loan_type_metrics = self._calculate_loan_type_metrics(df)

            # Process temporal patterns
            temporal_patterns = self._calculate_temporal_patterns(df)

            # Calculate success correlations
            amount_correlation = self._calculate_amount_success_correlation(df)

            return {
                'demographic_insights': self._extract_demographics(df),
                'loan_patterns': self._extract_loan_patterns(df),
                'performance_metrics': {
                    'overall_metrics': {
                        'total_applications': total_applications,
                        'status_distribution': status_distribution,
                        'total_amount': float(df['applied_amount'].sum()),
                        'average_amount': float(df['applied_amount'].mean())
                    },
                    'loan_type_metrics': loan_type_metrics,
                    'temporal_patterns': temporal_patterns,
                    'amount_success_correlation': amount_correlation
                }
            }
        except Exception as e:
            logger.error(f"Error in full dataset processing: {str(e)}")
            return self._get_empty_full_dataset_metrics()

    def _calculate_loan_type_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate loan type metrics with safe division handling"""
        loan_type_metrics = {}

        for loan_type in df['loan_type'].unique():
            try:
                loan_data = df[df['loan_type'] == loan_type]
                if len(loan_data) > 0:
                    approved = len(loan_data[loan_data['status'] == 'Approved'])
                    total = len(loan_data)
                    loan_type_metrics[loan_type] = {
                        'total_applications': total,
                        'approval_rate': self.safe_divide(approved, total),
                        'average_amount': float(loan_data['applied_amount'].mean()),
                        'total_amount': float(loan_data['applied_amount'].sum())
                    }
            except Exception as e:
                logger.warning(f"Error processing loan type {loan_type}: {str(e)}")
                continue

        return loan_type_metrics

    def _calculate_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Calculate temporal patterns with safe aggregation"""
        try:
            df['month'] = df['loan_start_date'].dt.to_period('M')

            # Group by month and calculate metrics
            temporal_data = df.groupby('month').agg({
                'application_id': 'count',
                'applied_amount': ['sum', 'mean'],
                'status': lambda x: self.safe_divide(
                    len(x[x == 'Approved']),
                    len(x)
                )
            }).tail(12)

            # Format results
            return {
                str(idx): {
                    'total_applications': int(row['application_id']['count']),
                    'total_amount': float(row['applied_amount']['sum']),
                    'avg_amount': float(row['applied_amount']['mean']),
                    'approval_rate': float(row['status'])
                }
                for idx, row in temporal_data.iterrows()
            }
        except Exception as e:
            logger.error(f"Error calculating temporal patterns: {str(e)}")
            return {}

    def _calculate_amount_success_correlation(self, df: pd.DataFrame) -> Dict:
        """Calculate amount success correlation with proper error handling"""
        try:
            if len(df) < 2:  # Need at least 2 points for correlation
                return {'amount_quartile_metrics': {}, 'correlation_coefficient': 0.0}

            # Create amount bins safely
            try:
                df['amount_quartile'] = pd.qcut(
                    df['applied_amount'],
                    q=4,
                    labels=['Q1', 'Q2', 'Q3', 'Q4']
                )
            except ValueError:
                # Handle case with too few unique values
                df['amount_quartile'] = pd.cut(
                    df['applied_amount'],
                    bins=4,
                    labels=['Q1', 'Q2', 'Q3', 'Q4']
                )

            # Calculate metrics for each quartile
            success_by_quartile = {}
            for quartile in df['amount_quartile'].dropna().unique():
                quartile_data = df[df['amount_quartile'] == quartile]
                if len(quartile_data) > 0:
                    success_rate = self.safe_divide(
                        len(quartile_data[quartile_data['status'] == 'Approved']),
                        len(quartile_data)
                    )

                    success_by_quartile[str(quartile)] = {
                        'success_rate': float(success_rate),
                        'total_applications': int(len(quartile_data)),
                        'amount_range': {
                            'min': float(quartile_data['applied_amount'].min()),
                            'max': float(quartile_data['applied_amount'].max())
                        }
                    }

            # Calculate correlation coefficient safely
            try:
                correlation = df['applied_amount'].corr(
                    (df['status'] == 'Approved').astype(float)
                )
                if pd.isna(correlation):
                    correlation = 0.0
            except Exception:
                correlation = 0.0

            return {
                'amount_quartile_metrics': success_by_quartile,
                'correlation_coefficient': float(correlation)
            }

        except Exception as e:
            logger.error(f"Error in amount correlation calculation: {str(e)}")
            return {
                'amount_quartile_metrics': {},
                'correlation_coefficient': 0.0
            }

    def _extract_demographics(self, df: pd.DataFrame) -> Dict:
        """Extract demographic insights with safe calculations"""
        try:
            total_applicants = len(df)
            if total_applicants == 0:
                return self._get_empty_demographics()

            # Calculate income distribution safely
            income_quartiles = df['income'].quantile([0.25, 0.5, 0.75])
            income_distribution = {
                'low': int(len(df[df['income'] <= income_quartiles[0.25]])),
                'medium': int(len(df[(df['income'] > income_quartiles[0.25]) &
                                     (df['income'] <= income_quartiles[0.75])])),
                'high': int(len(df[df['income'] > income_quartiles[0.75]]))
            }

            # Calculate temporal trends safely
            df['month_year'] = df['loan_start_date'].dt.to_period('M')
            temporal_trends = (df.groupby('month_year').size()
                               .sort_index()
                               .tail(12)
                               .to_dict())

            return {
                'total_applicants': total_applicants,
                'unique_customers': int(df['customer_id'].nunique()),
                'income_distribution': income_distribution,
                'occupation_distribution': df['occupation'].value_counts().to_dict(),
                'temporal_trends': {str(k): int(v) for k, v in temporal_trends.items()},
                'avg_income': float(df['income'].mean()),
                'median_income': float(df['income'].median())
            }

        except Exception as e:
            logger.error(f"Error in demographics extraction: {str(e)}")
            return self._get_empty_demographics()

    def _extract_loan_patterns(self, df: pd.DataFrame) -> Dict:
        """Extract loan patterns with safe calculations"""
        try:
            if len(df) == 0:
                return self._get_empty_loan_patterns()

            # Calculate quartiles safely
            quartiles = df['applied_amount'].quantile([0.25, 0.5, 0.75]).to_dict()

            # Process loan types with error handling
            loan_type_patterns = {}
            for loan_type in df['loan_type'].unique():
                try:
                    loan_data = df[df['loan_type'] == loan_type]
                    if len(loan_data) > 0:
                        loan_type_patterns[loan_type] = {
                            'avg_amount': float(loan_data['applied_amount'].mean()),
                            'median_amount': float(loan_data['applied_amount'].median()),
                            'count': int(len(loan_data)),
                            'total_amount': float(loan_data['applied_amount'].sum())
                        }
                except Exception:
                    continue

            # Calculate monthly trends safely
            monthly_data = (df.groupby(df['loan_start_date'].dt.to_period('M'))
                            .agg({
                'applied_amount': ['count', 'sum', 'mean']
            })
                            .round(2)
                            .tail(12))

            monthly_trends = {
                str(idx): {
                    'count': int(row['applied_amount']['count']),
                    'total_amount': float(row['applied_amount']['sum']),
                    'avg_amount': float(row['applied_amount']['mean'])
                }
                for idx, row in monthly_data.iterrows()
            }

            return {
                'loan_type_distribution': df['loan_type'].value_counts().to_dict(),
                'amount_metrics': {
                    'mean': float(df['applied_amount'].mean()),
                    'median': float(df['applied_amount'].median()),
                    'min': float(df['applied_amount'].min()),
                    'max': float(df['applied_amount'].max()),
                    'total': float(df['applied_amount'].sum()),
                    'quartiles': {str(k): float(v) for k, v in quartiles.items()}
                },
                'loan_type_patterns': loan_type_patterns,
                'monthly_trends': monthly_trends
            }

        except Exception as e:
            logger.error(f"Error in loan patterns extraction: {str(e)}")
            return self._get_empty_loan_patterns()

    @staticmethod
    def _convert_to_datetime(series: pd.Series) -> pd.Series:
        """Convert a series to datetime format with multiple format attempts"""
        date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d', '%d/%m/%Y', '%Y%m%d']

        for fmt in date_formats:
            try:
                converted = pd.to_datetime(series, format=fmt, errors='coerce')
                if not converted.isna().all():
                    return converted
            except Exception:
                continue

        # Try automatic parsing as last resort
        converted = pd.to_datetime(series, errors='coerce')
        if converted.isna().all():
            raise ValueError("Unable to parse dates in any recognized format")

        return converted

    def _process_small_dataset(self, df: pd.DataFrame) -> Dict:
        """Process small datasets with simplified metrics"""
        try:
            # Basic statistics that work well with small datasets
            income_stats = {
                'mean': float(df['income'].mean()) if not df['income'].empty else 0.0,
                'min': float(df['income'].min()) if not df['income'].empty else 0.0,
                'max': float(df['income'].max()) if not df['income'].empty else 0.0
            }

            # Calculate loan amounts safely
            amount_stats = {
                'mean': float(df['applied_amount'].mean()) if not df['applied_amount'].empty else 0.0,
                'min': float(df['applied_amount'].min()) if not df['applied_amount'].empty else 0.0,
                'max': float(df['applied_amount'].max()) if not df['applied_amount'].empty else 0.0,
                'total': float(df['applied_amount'].sum()) if not df['applied_amount'].empty else 0.0
            }

            # Get latest application details safely
            latest_idx = df['loan_start_date'].idxmax() if not df.empty else None
            latest_record = df.loc[latest_idx] if latest_idx is not None else None

            # Calculate approval metrics safely
            total_count = len(df)
            approved_count = len(df[df['status'] == 'Approved'])
            approval_rate = self.safe_divide(approved_count, total_count)

            return {
                'summary': {
                    'total_records': total_count,
                    'date_range': {
                        'start': str(df['loan_start_date'].min().date()) if not df.empty else None,
                        'end': str(df['loan_start_date'].max().date()) if not df.empty else None
                    },
                    'unique_customers': int(df['customer_id'].nunique())
                },
                'loan_metrics': {
                    'amount_stats': amount_stats,
                    'loan_types': df['loan_type'].value_counts().to_dict(),
                    'status_distribution': df['status'].value_counts().to_dict(),
                    'approval_rate': float(approval_rate)
                },
                'customer_insights': {
                    'income_stats': income_stats,
                    'occupation_distribution': df['occupation'].value_counts().to_dict(),
                    'average_loan_to_income': float(self.safe_divide(
                        df['applied_amount'].sum(),
                        df['income'].sum()
                    )) if not df.empty else 0.0
                },
                'latest_application': {
                    'date': str(latest_record['loan_start_date'].date()) if latest_record is not None else None,
                    'amount': float(latest_record['applied_amount']) if latest_record is not None else 0.0,
                    'status': latest_record['status'] if latest_record is not None else None
                },
                'data_quality': {
                    'completeness': {
                        col: float((~df[col].isna()).mean()) for col in df.columns
                    },
                    'processing_note': 'Processed using small dataset metrics'
                }
            }
        except Exception as e:
            logger.error(f"Error processing small dataset: {str(e)}")
            return self._get_empty_small_dataset_metrics()

    @staticmethod
    def _get_empty_demographics() -> Dict:
        """Return empty demographic structure"""
        return {
            'total_applicants': 0,
            'unique_customers': 0,
            'income_distribution': {'low': 0, 'medium': 0, 'high': 0},
            'occupation_distribution': {},
            'temporal_trends': {},
            'avg_income': 0.0,
            'median_income': 0.0
        }

    @staticmethod
    def _get_empty_loan_patterns() -> Dict:
        """Return empty loan patterns structure"""
        return {
            'loan_type_distribution': {},
            'amount_metrics': {
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'total': 0.0,
                'quartiles': {}
            },
            'loan_type_patterns': {},
            'monthly_trends': {}
        }

    @staticmethod
    def _get_empty_full_dataset_metrics() -> Dict:
        """Return empty metrics structure for full analysis"""
        return {
            'demographic_insights': {
                'total_applicants': 0,
                'unique_customers': 0,
                'income_distribution': {'low': 0, 'medium': 0, 'high': 0},
                'occupation_distribution': {},
                'temporal_trends': {},
                'avg_income': 0.0,
                'median_income': 0.0
            },
            'loan_patterns': {
                'loan_type_distribution': {},
                'amount_metrics': {
                    'mean': 0.0,
                    'median': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'total': 0.0,
                    'quartiles': {}
                },
                'loan_type_patterns': {},
                'monthly_trends': {}
            },
            'performance_metrics': {
                'overall_metrics': {
                    'total_applications': 0,
                    'status_distribution': {},
                    'total_amount': 0.0,
                    'average_amount': 0.0
                },
                'loan_type_metrics': {},
                'temporal_patterns': {},
                'amount_success_correlation': {
                    'amount_quartile_metrics': {},
                    'correlation_coefficient': 0.0
                }
            }
        }

    @staticmethod
    def _get_empty_small_dataset_metrics() -> Dict:
        """Return empty metrics structure for small datasets"""
        return {
            'summary': {
                'total_records': 0,
                'date_range': {'start': None, 'end': None},
                'unique_customers': 0
            },
            'loan_metrics': {
                'amount_stats': {
                    'mean': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'total': 0.0
                },
                'loan_types': {},
                'status_distribution': {},
                'approval_rate': 0.0
            },
            'customer_insights': {
                'income_stats': {'mean': 0.0, 'min': 0.0, 'max': 0.0},
                'occupation_distribution': {},
                'average_loan_to_income': 0.0
            },
            'latest_application': {
                'date': None,
                'amount': 0.0,
                'status': None
            },
            'data_quality': {
                'completeness': {},
                'processing_note': 'Empty dataset'
            }
        }
