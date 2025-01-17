# src/ml/openai_client.py
import openai
import json
from typing import Dict, List
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


def _create_strategy_prompt(analysis: Dict, pincode: str) -> str:
    """Create detailed prompt for strategy generation"""
    return f"""As an expert loan marketing strategist, analyze the following market data for pincode {pincode} 
    and provide detailed strategy recommendations:

    Market Analysis Data:
    {json.dumps(analysis, indent=2)}

    Based on this data, provide:
    1. Targeted marketing strategies for high-potential segments
    2. Channel recommendations with expected reach
    3. Product focus recommendations
    4. Risk mitigation approaches
    5. Implementation timeline

    Format the response as a JSON object with the following structure:
    {{
        "target_segments": [
            {{
                "segment": string,
                "potential": string,
                "marketing_approach": string,
                "expected_reach": number,
                "success_probability": float
            }}
        ],
        "channel_strategy": [
            {{
                "channel": string,
                "target_audience": string,
                "expected_reach": number,
                "cost_efficiency": string,
                "implementation_timeline": string
            }}
        ],
        "product_recommendations": [
            {{
                "product": string,
                "target_segment": string,
                "optimal_pricing": string,
                "unique_selling_points": [string]
            }}
        ],
        "risk_mitigation": [
            {{
                "risk_type": string,
                "severity": string,
                "mitigation_strategy": string,
                "action_items": [string]
            }}
        ],
        "implementation_plan": {{
            "phases": [
                {{
                    "phase": string,
                    "duration": string,
                    "key_activities": [string],
                    "expected_outcomes": {{
                        "reach": number,
                        "conversion": float
                    }}
                }}
            ],
            "success_metrics": {{
                "metric_name": {{
                    "target": number,
                    "timeline": string
                }}
            }}
        }}
    }}

    Ensure all recommendations are based on the provided market analysis data."""


def _validate_strategy(strategy: Dict) -> None:
    """Validate strategy response format"""
    required_keys = [
        'target_segments',
        'channel_strategy',
        'product_recommendations',
        'risk_mitigation',
        'implementation_plan'
    ]

    missing_keys = [key for key in required_keys if key not in strategy]
    if missing_keys:
        raise ValueError(f"Missing required strategy components: {missing_keys}")

    # Validate numeric values
    for segment in strategy.get('target_segments', []):
        if not isinstance(segment.get('expected_reach'), (int, float)):
            raise ValueError("Invalid expected_reach value in target_segments")
        if not isinstance(segment.get('success_probability'), float):
            raise ValueError("Invalid success_probability value in target_segments")


def _process_strategy_response(response: str) -> Dict:
    """Process and validate OpenAI response"""
    try:
        strategy = json.loads(response)
        _validate_strategy(strategy)
        return strategy
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing OpenAI response: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Invalid strategy format: {str(e)}")
        raise


class OpenAIStrategyGenerator:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4-1106-preview"

    async def generate_strategy(self, market_analysis: Dict, pincode: str) -> Dict:
        """Generate marketing strategy recommendations using OpenAI"""
        try:
            prompt = _create_strategy_prompt(market_analysis, pincode)
            response = await self._get_openai_response(prompt)
            return _process_strategy_response(response)
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            raise

    async def _get_openai_response(self, prompt: str) -> str:
        """Get response from OpenAI API"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert loan marketing strategist with deep understanding of:
                            1. Market segmentation and targeting
                            2. Channel optimization
                            3. Risk assessment and mitigation
                            4. Implementation planning"""
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

