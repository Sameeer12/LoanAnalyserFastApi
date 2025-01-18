import pandas as pd
from typing import Dict, List
import logging
import asyncio
from pathlib import Path
import yaml
import os
import openai
from faker import Faker
import random
from dotenv import load_dotenv

from data.data_processor import LoanDataProcessor
from analysis.market_analyzer import MarketAnalyzer
from src.ai.openai_client import OpenAIStrategyGenerator
from src.data.data_generator import DataGenerator

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class LoanStrategyApp:
    def __init__(self, config_path: str = "../config/config.yaml"):
        self.config = self._load_config(config_path)
        self.data_processor = LoanDataProcessor()
        self.market_analyzer = MarketAnalyzer()
        self.strategy_generator = OpenAIStrategyGenerator()

    def _load_config(self, config_path: str) -> Dict:
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

            return {
                'pincode': pincode,
                'analysis': pincode_data,
                'market_analysis': market_analysis,
                'strategy': strategy
            }

        except Exception as e:
            logger.error(f"Error analyzing pincode {pincode}: {str(e)}")
            raise


async def main():
    try:
        # Initialize application
        load_dotenv()
        app = LoanStrategyApp()
        data_generator = DataGenerator()
        # data_generator.generate_csv_data()
        # Load loan data
        logger.info("Loading loan data...")
        loan_data = pd.read_csv("data/loan_applications.csv")

        # Analyze specific pincode
        pincode = '110080'
        logger.info(f"Analyzing pincode {pincode}...")
        results = await app.analyze_pincode(loan_data, pincode)

        # Save results
        output_path = "output/analysis_results.yaml"
        logger.info(f"Saving results to {output_path}...")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"open api key: {api_key}")
    # models = openai.models.list()
    #
    # # Print the model names
    # for model in models['data']:
    #     print(model['id'])
    # print(f"open api key: {api_key}")
    asyncio.run(main())