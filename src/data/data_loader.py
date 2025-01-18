import pandas as pd
import yaml
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def load_loan_data(self, file_path: str) -> pd.DataFrame:
        """Load loan applications data"""
        try:
            df = pd.read_csv(file_path)
            df = self._preprocess_data(df)
            logger.info(f"Successfully loaded {len(df)} loan applications")
            return df
        except Exception as e:
            logger.error(f"Error loading loan data: {str(e)}")
            raise

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initial data preprocessing"""
        # Convert date columns
        date_format = self.config['data']['date_format']
        df['loan_start_date'] = pd.to_datetime(df['loan_start_date'], format=date_format)

        # Handle missing values
        numeric_fields = self.config['data']['numeric_fields']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')

        return df