# src/analysis/strategy_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class StrategyAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.lookback_days = config.get('analysis', {}).get('lookback_period_days', 90)
        self.min_applications = config.get('analysis', {}).get('min_applications', 50)

    def generate_strategy(self, df: pd.DataFrame, pincode: str) -> Dict:
        """Generate comprehensive strategy recommendations for a pincode"""
        try:
            # Get recent data for trend analysis
            recent_data = self._get_recent_data(df)

            return {
                'target_segments': self._identify_target_segments(df, recent_data),
                'product_strategies': self._develop_product_strategies(df, recent_data),
                'channel_recommendations': self._recommend_channels(df),
                'risk_strategies': self._analyze_risks(df),
                'implementation_plan': self._create_implementation_plan(df)
            }
        except Exception as e:
            logger.error(f"Error generating strategy for pincode {pincode}: {str(e)}")
            raise

    def _get_recent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get recent data based on lookback period"""
        cutoff_date = df['application_date'].max() - timedelta(days=self.lookback_days)
        return df[df['application_date'] > cutoff_date]

    def _identify_target_segments(self, df: pd.DataFrame, recent_data: pd.DataFrame) -> List[Dict]:
        """Identify and prioritize target segments"""
        segments = []

        # Analyze by occupation
        for occupation in df['occupation'].unique():
            segment_data = df[df['occupation'] == occupation]
            recent_segment = recent_data[recent_data['occupation'] == occupation]

            if len(segment_data) < self.min_applications:
                continue

            # Calculate segment metrics
            approved_mask = segment_data['status'] == 'Approved'
            approval_count = approved_mask.sum()
            total_count = len(segment_data)

            approval_rate = approval_count / total_count if total_count > 0 else 0
            avg_loan = segment_data['applied_amount'].mean() if not segment_data.empty else 0
            recent_growth = (len(recent_segment) / len(segment_data)) if len(segment_data) > 0 else 0

            # Calculate segment score
            segment_score = (approval_rate * 0.4 +
                             (avg_loan / df['applied_amount'].mean() if not df.empty else 0) * 0.3 +
                             recent_growth * 0.3)

            segments.append({
                'segment': occupation,
                'metrics': {
                    'approval_rate': approval_rate,
                    'avg_loan_amount': avg_loan,
                    'recent_growth': recent_growth,
                    'segment_score': segment_score
                },
                'recommendations': self._generate_segment_recommendations(
                    segment_data,
                    approval_rate,
                    avg_loan
                )
            })

        return sorted(segments, key=lambda x: x['metrics']['segment_score'], reverse=True)

    def _develop_product_strategies(self, df: pd.DataFrame, recent_data: pd.DataFrame) -> List[Dict]:
        """Develop strategies for different loan products"""
        products = []

        for loan_type in df['loan_type'].unique():
            product_data = df[df['loan_type'] == loan_type]
            recent_product = recent_data[recent_data['loan_type'] == loan_type]

            if len(product_data) < self.min_applications:
                continue

            # Calculate product metrics
            approved_mask = product_data['status'] == 'Approved'
            success_rate = approved_mask.sum() / len(product_data) if len(product_data) > 0 else 0
            avg_amount = product_data['applied_amount'].mean() if not product_data.empty else 0
            recent_trend = len(recent_product) / len(product_data) if len(product_data) > 0 else 0

            products.append({
                'loan_type': loan_type,
                'performance_metrics': {
                    'success_rate': success_rate,
                    'avg_amount': avg_amount,
                    'recent_trend': recent_trend
                },
                'recommendations': self._create_product_recommendations(
                    success_rate,
                    recent_trend,
                    avg_amount
                )
            })

        return sorted(products, key=lambda x: x['performance_metrics']['recent_trend'], reverse=True)

    @staticmethod
    def _create_product_recommendations(success_rate: float, recent_trend: float, avg_amount: float) -> List[Dict]:
        """Generate product-specific recommendations"""
        recommendations = []

        # Success rate based recommendations
        if success_rate > 0.7:
            recommendations.append({
                'type': 'Expansion',
                'priority': 'High',
                'actions': [
                    'Increase marketing budget',
                    'Expand target segments',
                    'Optimize pricing'
                ]
            })
        elif success_rate < 0.5:
            recommendations.append({
                'type': 'Optimization',
                'priority': 'High',
                'actions': [
                    'Review eligibility criteria',
                    'Enhance screening process',
                    'Improve documentation'
                ]
            })

        # Trend based recommendations
        if recent_trend > 1.2:
            recommendations.append({
                'type': 'Growth',
                'priority': 'Medium',
                'actions': [
                    'Scale operations',
                    'Enhance processing capacity',
                    'Develop upsell strategies'
                ]
            })
        elif recent_trend < 0.8:
            recommendations.append({
                'type': 'Revival',
                'priority': 'High',
                'actions': [
                    'Product enhancement',
                    'Competitive analysis',
                    'Market repositioning'
                ]
            })

        return recommendations

    @staticmethod
    def _generate_segment_recommendations(segment_data: pd.DataFrame,
                                          approval_rate: float,
                                          avg_loan: float) -> List[Dict]:
        """Generate segment-specific recommendations"""
        recommendations = []

        if approval_rate > 0.7:
            recommendations.append({
                'type': 'Expansion',
                'description': 'Increase market penetration',
                'actions': [
                    'Increase marketing spend',
                    'Expand channel presence',
                    'Develop targeted promotions'
                ]
            })
        elif 0.4 <= approval_rate <= 0.7:
            recommendations.append({
                'type': 'Optimization',
                'description': 'Improve conversion rates',
                'actions': [
                    'Enhance pre-screening',
                    'Optimize application process',
                    'Develop targeted products'
                ]
            })

        return recommendations

    def _recommend_channels(self, df: pd.DataFrame) -> List[Dict]:
        """Recommend marketing channels based on segment analysis"""
        channels = []

        # Calculate segment metrics
        segment_metrics = df.groupby('occupation').agg({
            'applied_amount': 'mean',
            'income': 'mean'
        }).reset_index()

        # Digital channels
        high_value_segments = segment_metrics[
            segment_metrics['applied_amount'] > segment_metrics['applied_amount'].median()
            ]
        if not high_value_segments.empty:
            channels.append({
                'channel': 'Digital',
                'target_segments': high_value_segments['occupation'].tolist(),
                'expected_reach': len(df[df['occupation'].isin(high_value_segments['occupation'])]),
                'strategy': 'Focus on mobile and online platforms'
            })

        # Traditional channels
        other_segments = segment_metrics[
            segment_metrics['applied_amount'] <= segment_metrics['applied_amount'].median()
            ]
        if not other_segments.empty:
            channels.append({
                'channel': 'Traditional',
                'target_segments': other_segments['occupation'].tolist(),
                'expected_reach': len(df[df['occupation'].isin(other_segments['occupation'])]),
                'strategy': 'Focus on direct outreach and local presence'
            })

        return channels

    def _analyze_risks(self, df: pd.DataFrame) -> Dict:
        """Analyze and assess risks"""
        return {
            'portfolio_concentration': self._calculate_concentration_risk(df),
            'segment_risks': self._calculate_segment_risks(df),
            'mitigation_strategies': self._create_risk_mitigation_strategies(df)
        }

    def _calculate_concentration_risk(self, df: pd.DataFrame) -> Dict:
        """Calculate concentration risk metrics"""
        return {
            'product_concentration': df['loan_type'].value_counts(normalize=True).to_dict(),
            'segment_concentration': df['occupation'].value_counts(normalize=True).to_dict()
        }

    def _calculate_segment_risks(self, df: pd.DataFrame) -> Dict:
        """Calculate segment-specific risks"""
        segment_risks = {}

        for occupation in df['occupation'].unique():
            segment_data = df[df['occupation'] == occupation]
            approved_mask = segment_data['status'] == 'Approved'

            segment_risks[occupation] = {
                'rejection_rate': 1 - (approved_mask.sum() / len(segment_data)) if len(segment_data) > 0 else 0,
                'avg_loan': segment_data['applied_amount'].mean() if not segment_data.empty else 0
            }

        return segment_risks

    def _create_risk_mitigation_strategies(self, df: pd.DataFrame) -> List[Dict]:
        """Create risk mitigation strategies"""
        strategies = []

        # High rejection rate segments
        high_risk_segments = []
        for occupation in df['occupation'].unique():
            segment_data = df[df['occupation'] == occupation]
            if len(segment_data) > 0:
                rejection_rate = 1 - (segment_data['status'] == 'Approved').mean()
                if rejection_rate > 0.5:
                    high_risk_segments.append(occupation)

        if high_risk_segments:
            strategies.append({
                'risk_type': 'High Rejection Rate',
                'segments': high_risk_segments,
                'actions': [
                    'Enhance screening process',
                    'Review eligibility criteria',
                    'Implement segment-specific scoring'
                ]
            })

        return strategies

    def _create_implementation_plan(self, df: pd.DataFrame) -> Dict:
        """Create implementation plan"""
        return {
            'phases': [
                {
                    'phase': 'Initial',
                    'duration': '4 weeks',
                    'activities': [
                        'Setup targeting framework',
                        'Channel optimization',
                        'Risk controls implementation'
                    ]
                },
                {
                    'phase': 'Growth',
                    'duration': '8 weeks',
                    'activities': [
                        'Scale successful channels',
                        'Expand target segments',
                        'Optimize conversion funnel'
                    ]
                }
            ],
            'metrics': {
                'target_conversion': 0.15,
                'target_growth': 0.25
            }
        }