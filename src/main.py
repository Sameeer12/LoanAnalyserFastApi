# import uvicorn
# from src.api.routes import app  # Replace this with the actual import path to your FastAPI app instance
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


# src/main.py
import pandas as pd
from typing import Dict, List
import logging
import asyncio
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

from data.data_processor import LoanDataProcessor
from analysis.market_analyzer import MarketAnalyzer
from src.ai.openai_client import OpenAIStrategyGenerator

# Load environment variables
load_dotenv()


class LoanStrategyApp:
    def __init__(self, config_path: str = "../config/config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()

        # Initialize components
        self.data_processor = LoanDataProcessor()
        self.market_analyzer = MarketAnalyzer()
        self.strategy_generator = OpenAIStrategyGenerator(
            api_key="sk-w2vmuOkV3gXjPNNzs74c4-cMOWxRp7UezE__QYNXP8T3BlbkFJ7HqG4zGifgy-AaPyIGdELNddadbn1WlqMlkcBEmIMA")

        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=self.config['logging']['level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def analyze_pincode(self, loan_data: pd.DataFrame, pincode: str) -> Dict:
        """Analyze and generate strategy for a specific pincode"""
        try:
            # Process loan data
            self.logger.info(f"Processing data for pincode {pincode}")
            processed_data = self.data_processor.process_loan_data(loan_data)
            pincode_data = processed_data.get(pincode)

            if not pincode_data:
                raise ValueError(f"No data available for pincode {pincode}")

            # Analyze market potential
            self.logger.info("Analyzing market potential")
            market_analysis = self.market_analyzer.analyze_market_potential(
                loan_data[loan_data['pincode'] == pincode],
                pincode
            )

            # Generate AI-powered strategy
            self.logger.info("Generating strategy recommendations")
            strategy = await self.strategy_generator.generate_strategy(
                market_analysis,
                pincode
            )

            return {
                'pincode': pincode,
                'market_analysis': market_analysis,
                'strategy_recommendations': strategy
            }

        except Exception as e:
            self.logger.error(f"Error analyzing pincode {pincode}: {str(e)}")
            raise

    async def batch_analyze_pincodes(self, loan_data: pd.DataFrame,
                                     pincodes: List[str] = None) -> Dict[str, Dict]:
        """Analyze multiple pincodes"""
        if pincodes is None:
            pincodes = loan_data['pincode'].unique()

        results = {}
        for pincode in pincodes:
            try:
                results[pincode] = await self.analyze_pincode(loan_data, pincode)
                self.logger.info(f"Completed analysis for pincode {pincode}")
            except Exception as e:
                self.logger.error(f"Error analyzing pincode {pincode}: {str(e)}")
                results[pincode] = {'error': str(e)}

        return results

    def save_results(self, results: Dict, output_path: str) -> None:
        """Save analysis results to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)

            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise


async def main():
    # Initialize application
    app = LoanStrategyApp()

    # Load loan data
    loan_data = pd.read_csv("data/loan_applications.csv")

    # Analyze specific pincodes
    pincodes_to_analyze = ['110001', '400001']  # Example pincodes
    results = await app.batch_analyze_pincodes(loan_data, pincodes_to_analyze)

    # Save results
    app.save_results(results, "output/analysis_results.yaml")


if __name__ == "__main__":
    asyncio.run(main())