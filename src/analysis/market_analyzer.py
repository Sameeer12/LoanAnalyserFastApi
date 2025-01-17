# src/analysis/market_analyzer.py
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    def __init__(self):
        self.lookback_days = 90  # Analysis window for recent trends

    def analyze_market_potential(self, df: pd.DataFrame, pincode: str) -> Dict:
        """Analyze market potential for a specific pincode"""
        try:
            recent_data = self._get_recent_data(df)

            return {
                'market_size': self._analyze_market_size(df, recent_data),
                'growth_patterns': self._analyze_growth_patterns(df, recent_data),
                'segment_opportunities': self._analyze_segment_opportunities(df),
                'risk_assessment': self._assess_risks(df)
            }
        except Exception as e:
            logger.error(f"Error in market analysis for pincode {pincode}: {str(e)}")
            raise

    def _get_recent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get recent data for trend analysis"""
        cutoff_date = df['application_date'].max() - timedelta(days=self.lookback_days)
        return df[df['application_date'] > cutoff_date]

    def _analyze_market_size(self, df: pd.DataFrame, recent_data: pd.DataFrame) -> Dict:
        """Analyze current and potential market size"""
        return {
            'current_metrics': {
                'total_applications': len(df),
                'total_customers': df['customer_id'].nunique(),
                'total_approved_value': df[df['status'] == 'Approved']['applied_amount'].sum()
            },
            'recent_trends': {
                'application_volume': len(recent_data),
                'approval_rate': len(recent_data[recent_data['status'] == 'Approved']) / len(recent_data),
                'avg_ticket_size': recent_data['applied_amount'].mean()
            }
        }

    def _analyze_growth_patterns(self, df: pd.DataFrame, recent_data: pd.DataFrame) -> Dict:
        """Analyze growth patterns in different segments"""
        # Calculate monthly trends
        df['month'] = df['application_date'].dt.to_period('M')
        monthly_volumes = df.groupby('month').size()

        # Calculate growth rates
        if len(monthly_volumes) >= 2:
            growth_rate = (monthly_volumes.iloc[-1] / monthly_volumes.iloc[-2]) - 1
        else:
            growth_rate = 0

        return {
            'monthly_growth_rate': growth_rate,
            'loan_type_growth': self._calculate_type_growth(df, recent_data),
            'segment_growth': self._calculate_segment_growth(df, recent_data)
        }

    def _calculate_type_growth(self, df: pd.DataFrame, recent_data: pd.DataFrame) -> Dict:
        """Calculate growth rates by loan type"""
        growth_rates = {}
        for loan_type in df['loan_type'].unique():
            historical_share = len(df[df['loan_type'] == loan_type]) / len(df)
            recent_share = len(recent_data[recent_data['loan_type'] == loan_type]) / len(recent_data)
            growth_rates[loan_type] = {
                'share_change': recent_share - historical_share,
                'approval_rate': len(recent_data[(recent_data['loan_type'] == loan_type) &
                                                 (recent_data['status'] == 'Approved')]) / len(
                    recent_data[recent_data['loan_type'] == loan_type])
            }
        return growth_rates

    def _calculate_segment_growth(self, df: pd.DataFrame, recent_data: pd.DataFrame) -> Dict:
        """Calculate growth rates by customer segment"""
        growth_rates = {}
        for occupation in df['occupation'].unique():
            historical_share = len(df[df['occupation'] == occupation]) / len(df)
            recent_share = len(recent_data[recent_data['occupation'] == occupation]) / len(recent_data)
            growth_rates[occupation] = {
                'share_change': recent_share - historical_share,
                'avg_loan_amount': recent_data[recent_data['occupation'] == occupation]['applied_amount'].mean()
            }
        return growth_rates

    def _analyze_segment_opportunities(self, df: pd.DataFrame) -> Dict:
        """Identify high-potential market segments"""
        segment_metrics = {}

        for occupation in df['occupation'].unique():
            segment_data = df[df['occupation'] == occupation]
            approval_rate = len(segment_data[segment_data['status'] == 'Approved']) / len(segment_data)
            avg_amount = segment_data['applied_amount'].mean()

            segment_metrics[occupation] = {
                'size': len(segment_data),
                'approval_rate': approval_rate,
                'avg_loan_amount': avg_amount,
                'potential_score': approval_rate * (avg_amount / df['applied_amount'].mean())
            }

        # Identify high-potential segments
        high_potential = {
            segment: metrics for segment, metrics in segment_metrics.items()
            if metrics['potential_score'] > 1.0 and metrics['approval_rate'] > 0.6
        }

        return {
            'segment_metrics': segment_metrics,
            'high_potential_segments': high_potential
        }

    def _assess_risks(self, df: pd.DataFrame) -> Dict:
        """Assess market risks and concentration"""
        # Calculate concentration metrics
        type_concentration = self._calculate_concentration(df['loan_type'])
        occupation_concentration = self._calculate_concentration(df['occupation'])

        # Calculate volatility
        monthly_volumes = df.groupby(df['application_date'].dt.to_period('M')).size()
        volatility = monthly_volumes.std() / monthly_volumes.mean() if len(monthly_volumes) > 1 else 0

        return {
            'concentration_risk': {
                'loan_type_concentration': type_concentration,
                'occupation_concentration': occupation_concentration
            },
            'volatility': volatility,
            'rejection_analysis': self._analyze_rejections(df)
        }

    def _calculate_concentration(self, series: pd.Series) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        proportions = series.value_counts(normalize=True)
        return (proportions ** 2).sum()

    # def _analyze_rejections(self, df: pd.DataFrame) -> Dict:
    #     """Analyze rejection patterns"""
    #     rejected = df[df['status'] == 'Rejected']
    #
    #     return {
    #         'overall_rejection_rate': len(rejected) / len(df),
    #         'rejection_by_loan_type': {
    #             loan_type: len(rejected[rejected['loan_type'] == loan_type]) / len(df[df['loan_type'] == loan_type])
    #             for loan_type in df['loan_type'].unique()
    #         },
    #         'high_risk_segments': [
    #             occupation for occupation in df['occupation'].unique()
    #             if len(rejected[rejected['occupation'] == occupation]) / len(

        # Continuing src/analysis/market_analyzer.py
        def _analyze_rejections(self, df: pd.DataFrame) -> Dict:
            """Analyze rejection patterns"""
            rejected = df[df['status'] == 'Rejected']

            high_risk_segments = []
            for occupation in df['occupation'].unique():
                segment_data = df[df['occupation'] == occupation]
                rejection_rate = len(rejected[rejected['occupation'] == occupation]) / len(segment_data)
                if rejection_rate > 0.5:  # More than 50% rejection rate
                    high_risk_segments.append({
                        'segment': occupation,
                        'rejection_rate': rejection_rate,
                        'avg_income': segment_data['income'].mean(),
                        'avg_loan_amount': segment_data['applied_amount'].mean()
                    })

            return {
                'overall_rejection_rate': len(rejected) / len(df),
                'rejection_by_loan_type': {
                    loan_type: len(rejected[rejected['loan_type'] == loan_type]) /
                               len(df[df['loan_type'] == loan_type])
                    for loan_type in df['loan_type'].unique()
                },
                'high_risk_segments': high_risk_segments
            }

        def generate_recommendations(self, analysis_results: Dict) -> Dict:
            """Generate market strategy recommendations based on analysis"""
            recommendations = {
                'priority_segments': self._identify_priority_segments(analysis_results),
                'growth_opportunities': self._identify_growth_opportunities(analysis_results),
                'risk_mitigation': self._generate_risk_mitigation_strategies(analysis_results)
            }
            return recommendations

        def _identify_priority_segments(self, analysis: Dict) -> List[Dict]:
            """Identify priority segments for targeting"""
            segment_opportunities = analysis['segment_opportunities']
            high_potential = segment_opportunities['high_potential_segments']

            priority_segments = []
            for segment, metrics in high_potential.items():
                priority_segments.append({
                    'segment': segment,
                    'potential_score': metrics['potential_score'],
                    'recommended_approach': self._generate_segment_approach(metrics),
                    'target_products': self._identify_suitable_products(metrics)
                })

            return sorted(priority_segments, key=lambda x: x['potential_score'], reverse=True)

        def _generate_segment_approach(self, metrics: Dict) -> Dict:
            """Generate targeting approach for a segment"""
            return {
                'focus_areas': [
                    'Digital marketing' if metrics['approval_rate'] > 0.7 else 'Targeted outreach',
                    'Value proposition' if metrics['avg_loan_amount'] > 50000 else 'Volume focus'
                ],
                'key_considerations': [
                    f"Historical approval rate: {metrics['approval_rate']:.1%}",
                    f"Average loan size: ₹{metrics['avg_loan_amount']:,.0f}"
                ]
            }

        def _identify_suitable_products(self, metrics: Dict) -> List[Dict]:
            """Identify suitable loan products for segment"""
            products = []
            if metrics['avg_loan_amount'] > 100000:
                products.append({
                    'type': 'Business Loan',
                    'target_amount': metrics['avg_loan_amount'],
                    'success_probability': metrics['approval_rate']
                })
            if 30000 <= metrics['avg_loan_amount'] <= 100000:
                products.append({
                    'type': 'Personal Loan',
                    'target_amount': metrics['avg_loan_amount'],
                    'success_probability': metrics['approval_rate']
                })
            return products

        def _identify_growth_opportunities(self, analysis: Dict) -> List[Dict]:
            """Identify growth opportunities in the market"""
            growth_patterns = analysis['growth_patterns']

            opportunities = []
            for loan_type, metrics in growth_patterns['loan_type_growth'].items():
                if metrics['share_change'] > 0 and metrics['approval_rate'] > 0.6:
                    opportunities.append({
                        'product': loan_type,
                        'growth_rate': metrics['share_change'],
                        'success_rate': metrics['approval_rate'],
                        'recommendation': self._generate_growth_recommendation(metrics)
                    })

            return sorted(opportunities, key=lambda x: x['growth_rate'], reverse=True)

        def _generate_growth_recommendation(self, metrics: Dict) -> Dict:
            """Generate specific growth recommendations"""
            return {
                'strategy': 'Expansion' if metrics['share_change'] > 0.1 else 'Optimization',
                'focus_areas': [
                    'Market penetration' if metrics['approval_rate'] > 0.7 else 'Risk optimization',
                    'Volume growth' if metrics['share_change'] > 0.15 else 'Quality focus'
                ]
            }

        def _generate_risk_mitigation_strategies(self, analysis: Dict) -> List[Dict]:
            """Generate risk mitigation strategies"""
            risk_assessment = analysis['risk_assessment']

            strategies = []
            # Concentration risk strategies
            if risk_assessment['concentration_risk']['loan_type_concentration'] > 0.3:
                strategies.append({
                    'risk_type': 'Concentration Risk',
                    'severity': 'High',
                    'mitigation_strategy': 'Portfolio diversification',
                    'action_items': [
                        'Expand product offerings',
                        'Balance portfolio allocation',
                        'Develop new market segments'
                    ]
                })

            # Volatility risk strategies
            if risk_assessment['volatility'] > 0.2:
                strategies.append({
                    'risk_type': 'Market Volatility',
                    'severity': 'Medium',
                    'mitigation_strategy': 'Stabilization measures',
                    'action_items': [
                        'Implement counter-cyclical measures',
                        'Develop stable customer segments',
                        'Optimize marketing timing'
                    ]
                })

            # Rejection risk strategies
            high_rejection_segments = [
                segment for segment in risk_assessment['rejection_analysis']['high_risk_segments']
                if segment['rejection_rate'] > 0.5
            ]
            if high_rejection_segments:
                strategies.append({
                    'risk_type': 'High Rejection Rate',
                    'severity': 'Medium',
                    'mitigation_strategy': 'Application quality improvement',
                    'action_items': [
                        'Enhance pre-screening process',
                        'Improve application guidance',
                        'Develop targeted eligibility criteria'
                    ]
                })

            return strategies