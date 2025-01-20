import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from src.ai.openai_client import OpenAIStrategyGenerator
from src.analysis.market_analyzer import MarketAnalyzer
from src.data.data_processor import LoanDataProcessor

logger = logging.getLogger(__name__)


# Load config
# config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
# with open(config_path, 'r') as f:
#     config = yaml.safe_load(f)

class LoanStrategyApp:
    def __init__(self, config_path=Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'):
        self.config = self._load_config(config_path)
        self.data_processor = LoanDataProcessor()
        self.market_analyzer = MarketAnalyzer()
        self.strategy_generator = OpenAIStrategyGenerator()

    def _load_config(self, config_path) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    async def analyze_pincode(self, loan_data: pd.DataFrame, pincode: str) -> Dict:
        """Analyze and generate strategy for a specific pincode"""
        try:
            # Convert pincode to string to ensure matching
            pincode = str(pincode)
            logger.info(f"Processing data for pincode {pincode}")

            # Process all loan data first
            processed_data = self.data_processor.process_loan_data(loan_data)
            pincode_data = processed_data.get(pincode)

            if not pincode_data:
                logger.info(f"No data available for pincode {pincode}")
                # raise ValueError(f"No data available for pincode {pincode}")

            # Analyze market potential
            logger.info("Analyzing market potential")
            filtered_data = loan_data[loan_data['pincode'].astype(str) == pincode].copy()
            market_analysis = self.market_analyzer.analyze_market_potential(
                filtered_data,
                pincode
            )

            # Generate strategy
            logger.info("Generating strategy recommendations")
            strategy = await self.strategy_generator.generate_strategy(
                market_analysis,
                pincode
            )
            strategy_recommendations = strategy or {"message": "No strategy recommendations available"}

            return {
                'pincode': pincode,
                'analysis': pincode_data,
                'market_analysis': market_analysis,
                'strategy_recommendations': strategy_recommendations
            }

        except Exception as e:
            logger.error(f"Error analyzing pincode {pincode}: {str(e)}")
            raise
