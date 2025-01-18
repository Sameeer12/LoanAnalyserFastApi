import openai
import json
from typing import Dict
import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

# Load the .env file
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    raise FileNotFoundError(f".env file not found at {dotenv_path}")

# Access environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAIStrategyGenerator:
    def __init__(self):
        self.model = "gpt-4o-mini"
        # self.client = openai.Client()

    async def generate_strategy(self, market_analysis: Dict, pincode: str) -> Dict:
        """Generate marketing strategy recommendations using OpenAI."""
        try:
            print("open api key: {openai.api_key}")
            prompt = self._create_strategy_prompt(market_analysis, pincode)
            response = await self._get_openai_response(prompt)
            return self._process_strategy_response(response)
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            return {}

    async def _get_openai_response(self, prompt: str) -> str:
        """Get response from OpenAI API."""
        try:
            response = await openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert loan marketing strategist with deep understanding of:\n"
                            "1. Market segmentation and targeting\n"
                            "2. Channel optimization\n"
                            "3. Risk assessment and mitigation\n"
                            "4. Implementation planning"
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return ""

    def _create_strategy_prompt(self, analysis: Dict, pincode: str) -> str:
        """Create a detailed prompt for strategy generation."""
        return (
            f"As an expert loan marketing strategist, analyze the following market data for pincode {pincode} "
            "and provide detailed strategy recommendations:\n\n"
            f"Market Analysis Data:\n{json.dumps(analysis, indent=2)}\n\n"
            "Based on this data, provide:\n"
            "1. Targeted marketing strategies for high-potential segments\n"
            "2. Channel recommendations with expected reach\n"
            "3. Product focus recommendations\n"
            "4. Risk mitigation approaches\n"
            "5. Implementation timeline\n\n"
            "Format the response as a JSON object with the following structure:\n"
            "{\n"
            '    "target_segments": [\n'
            '        {"segment": string, "potential": string, "marketing_approach": string, "expected_reach": number, '
            '"success_probability": float}\n'
            "    ],\n"
            '    "channel_strategy": [\n'
            '        {"channel": string, "target_audience": string, "expected_reach": number, "cost_efficiency": string, '
            '"implementation_timeline": string}\n'
            "    ],\n"
            '    "product_recommendations": [\n'
            '        {"product": string, "target_segment": string, "optimal_pricing": string, "unique_selling_points": [string]}\n'
            "    ],\n"
            '    "risk_mitigation": [\n'
            '        {"risk_type": string, "severity": string, "mitigation_strategy": string, "action_items": [string]}\n'
            "    ],\n"
            '    "implementation_plan": {\n'
            '        "phases": [\n'
            '            {"phase": string, "duration": string, "key_activities": [string], "expected_outcomes": {"reach": number, '
            '"conversion": float}}\n'
            "        ],\n"
            '        "success_metrics": {"metric_name": {"target": number, "timeline": string}}\n'
            "    }\n"
            "}\n"
            "Ensure all recommendations are based on the provided market analysis data."
        )

    def _validate_strategy(self, strategy: Dict) -> None:
        """Validate strategy response format."""
        required_keys = [
            "target_segments",
            "channel_strategy",
            "product_recommendations",
            "risk_mitigation",
            "implementation_plan"
        ]

        missing_keys = [key for key in required_keys if key not in strategy]
        if missing_keys:
            logger.warning(f"Missing required strategy components: {missing_keys}")

        # Validate numeric values and log warnings for invalid data
        for segment in strategy.get("target_segments", []):
            if not isinstance(segment.get("expected_reach"), (int, float)):
                logger.warning("Invalid 'expected_reach' value in target_segments")
            if not isinstance(segment.get("success_probability"), float):
                logger.warning("Invalid 'success_probability' value in target_segments")

    def _process_strategy_response(self, response: str) -> Dict:
        """Process and validate OpenAI response."""
        if not response:
            logger.warning("Empty response from OpenAI.")
            return {}

        try:
            strategy = json.loads(response)
            self._validate_strategy(strategy)
            return strategy
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            return {}