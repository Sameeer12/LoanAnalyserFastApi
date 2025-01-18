import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    def __init__(self):
        self.lookback_days = 3650  # Analysis window for recent trends

    def analyze_market_potential(self, df: pd.DataFrame, pincode: str) -> Dict:
        """Analyze market potential for a specific pincode"""
        try:
            # Ensure datetime conversion first
            df = self._prepare_data(df)
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

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by ensuring proper types and formats"""
        try:
            df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df['loan_start_date']):
                logger.info("Converting loan_start_date to datetime")
                df['loan_start_date'] = pd.to_datetime(df['loan_start_date'])
            return df
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def _get_recent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get recent data for trend analysis"""
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided")
                return df

            max_date = df['loan_start_date'].max()
            cutoff_date = max_date - timedelta(days=self.lookback_days)
            recent_data = df[df['loan_start_date'] > cutoff_date].copy()
            logger.info(f"Filtered data from {cutoff_date} to {max_date}")

            return recent_data
        except Exception as e:
            logger.error(f"Error getting recent data: {str(e)}")
            raise

    def _analyze_market_size(self, df: pd.DataFrame, recent_data: pd.DataFrame) -> Dict:
        """Analyze current and potential market size"""
        try:
            return {
                'current_metrics': {
                    'total_applications': int(len(df)),
                    'total_customers': int(df['customer_id'].nunique()),
                    'total_approved_value': float(df[df['status'] == 'Approved']['applied_amount'].sum())
                },
                'recent_trends': {
                    'application_volume': int(len(recent_data)),
                    'approval_rate': float(
                        len(recent_data[recent_data['status'] == 'Approved']) / len(recent_data)) if len(
                        recent_data) > 0 else 0.0,
                    'avg_ticket_size': float(recent_data['applied_amount'].mean()) if len(recent_data) > 0 else 0.0
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing market size: {str(e)}")
            raise

    def _analyze_growth_patterns(self, df: pd.DataFrame, recent_data: pd.DataFrame) -> Dict:
        """Analyze growth patterns in different segments"""
        try:
            monthly_volumes = df.groupby(df['loan_start_date'].dt.to_period('M')).size()
            growth_rate = 0.0
            if len(monthly_volumes) >= 2:
                growth_rate = float((monthly_volumes.iloc[-1] / monthly_volumes.iloc[-2]) - 1)

            return {
                'monthly_growth_rate': growth_rate,
                'loan_type_growth': self._calculate_type_growth(df, recent_data),
                'segment_growth': self._calculate_segment_growth(df, recent_data)
            }
        except Exception as e:
            logger.error(f"Error analyzing growth patterns: {str(e)}")
            raise

    def _calculate_type_growth(self, df: pd.DataFrame, recent_data: pd.DataFrame) -> Dict:
        """Calculate growth rates by loan type"""
        growth_rates = {}
        try:
            for loan_type in df['loan_type'].unique():
                historical_share = len(df[df['loan_type'] == loan_type]) / len(df) if len(df) > 0 else 0
                recent_share = (len(recent_data[recent_data['loan_type'] == loan_type]) /
                                len(recent_data)) if len(recent_data) > 0 else 0

                recent_type_data = recent_data[recent_data['loan_type'] == loan_type]
                approval_rate = (
                        len(recent_type_data[recent_type_data['status'] == 'Approved']) /
                        len(recent_type_data)
                ) if len(recent_type_data) > 0 else 0

                growth_rates[loan_type] = {
                    'share_change': float(recent_share - historical_share),
                    'approval_rate': float(approval_rate)
                }
            return growth_rates
        except Exception as e:
            logger.error(f"Error calculating type growth: {str(e)}")
            raise

    def _calculate_segment_growth(self, df: pd.DataFrame, recent_data: pd.DataFrame) -> Dict:
        """Calculate growth rates by customer segment"""
        try:
            growth_rates = {}
            for occupation in df['occupation'].unique():
                historical_share = len(df[df['occupation'] == occupation]) / len(df) if len(df) > 0 else 0
                recent_share = (len(recent_data[recent_data['occupation'] == occupation]) /
                                len(recent_data)) if len(recent_data) > 0 else 0

                recent_segment_data = recent_data[recent_data['occupation'] == occupation]
                avg_amount = float(recent_segment_data['applied_amount'].mean()) if len(
                    recent_segment_data) > 0 else 0.0

                growth_rates[occupation] = {
                    'share_change': float(recent_share - historical_share),
                    'avg_loan_amount': avg_amount
                }
            return growth_rates
        except Exception as e:
            logger.error(f"Error calculating segment growth: {str(e)}")
            raise

    def _analyze_segment_opportunities(self, df: pd.DataFrame) -> Dict:
        """Identify high-potential market segments"""
        try:
            segment_metrics = {}
            overall_mean_amount = float(df['applied_amount'].mean()) if len(df) > 0 else 0.0

            for occupation in df['occupation'].unique():
                segment_data = df[df['occupation'] == occupation]
                segment_size = len(segment_data)

                if segment_size > 0:
                    approval_rate = len(segment_data[segment_data['status'] == 'Approved']) / segment_size
                    avg_amount = float(segment_data['applied_amount'].mean())

                    potential_score = approval_rate * (
                                avg_amount / overall_mean_amount) if overall_mean_amount > 0 else 0.0

                    segment_metrics[occupation] = {
                        'size': int(segment_size),
                        'approval_rate': float(approval_rate),
                        'avg_loan_amount': float(avg_amount),
                        'potential_score': float(potential_score)
                    }

            high_potential = {
                segment: metrics for segment, metrics in segment_metrics.items()
                if metrics['potential_score'] > 1.0 and metrics['approval_rate'] > 0.6
            }

            return {
                'segment_metrics': segment_metrics,
                'high_potential_segments': high_potential
            }
        except Exception as e:
            logger.error(f"Error analyzing segment opportunities: {str(e)}")
            raise

    def _assess_risks(self, df: pd.DataFrame) -> Dict:
        """Assess market risks and concentration"""
        try:
            type_concentration = self._calculate_concentration(df['loan_type'])
            occupation_concentration = self._calculate_concentration(df['occupation'])

            monthly_volumes = df.groupby(df['loan_start_date'].dt.to_period('M')).size()
            volatility = float(monthly_volumes.std() / monthly_volumes.mean()) if len(monthly_volumes) > 1 else 0.0

            return {
                'concentration_risk': {
                    'loan_type_concentration': float(type_concentration),
                    'occupation_concentration': float(occupation_concentration)
                },
                'volatility': volatility,
                'rejection_analysis': self._analyze_rejections(df)
            }
        except Exception as e:
            logger.error(f"Error assessing risks: {str(e)}")
            raise

    def _calculate_concentration(self, series: pd.Series) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        try:
            proportions = series.value_counts(normalize=True)
            return float((proportions ** 2).sum())
        except Exception as e:
            logger.error(f"Error calculating concentration: {str(e)}")
            return 0.0

    def _analyze_rejections(self, df: pd.DataFrame) -> Dict:
        """Analyze rejection patterns"""
        try:
            rejected = df[df['status'] == 'Rejected']
            total_count = len(df)

            high_risk_segments = []
            for occupation in df['occupation'].unique():
                segment_data = df[df['occupation'] == occupation]
                segment_count = len(segment_data)

                if segment_count > 0:
                    rejection_rate = len(rejected[rejected['occupation'] == occupation]) / segment_count
                    if rejection_rate > 0.5:
                        high_risk_segments.append({
                            'segment': occupation,
                            'rejection_rate': float(rejection_rate),
                            'avg_income': float(segment_data['income'].mean()),
                            'avg_loan_amount': float(segment_data['applied_amount'].mean())
                        })

            rejection_by_type = {}
            for loan_type in df['loan_type'].unique():
                type_data = df[df['loan_type'] == loan_type]
                type_count = len(type_data)
                rejection_by_type[loan_type] = float(
                    len(rejected[rejected['loan_type'] == loan_type]) / type_count
                ) if type_count > 0 else 0.0

            return {
                'overall_rejection_rate': float(len(rejected) / total_count) if total_count > 0 else 0.0,
                'rejection_by_loan_type': rejection_by_type,
                'high_risk_segments': high_risk_segments
            }
        except Exception as e:
            logger.error(f"Error analyzing rejections: {str(e)}")
            raise

    def generate_recommendations(self, analysis_results: Dict) -> Dict:
        """Generate market strategy recommendations based on analysis"""
        try:
            return {
                'priority_segments': self._identify_priority_segments(analysis_results),
                'growth_opportunities': self._identify_growth_opportunities(analysis_results),
                'risk_mitigation': self._generate_risk_mitigation_strategies(analysis_results)
            }
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise

    def _identify_priority_segments(self, analysis: Dict) -> List[Dict]:
        """Identify priority segments for targeting"""
        try:
            segment_opportunities = analysis['segment_opportunities']
            high_potential = segment_opportunities['high_potential_segments']

            priority_segments = []
            for segment, metrics in high_potential.items():
                priority_segments.append({
                    'segment': segment,
                    'potential_score': float(metrics['potential_score']),
                    'recommended_approach': self._generate_segment_approach(metrics),
                    'target_products': self._identify_suitable_products(metrics)
                })

            return sorted(priority_segments, key=lambda x: x['potential_score'], reverse=True)
        except Exception as e:
            logger.error(f"Error identifying priority segments: {str(e)}")
            raise

    def _generate_segment_approach(self, metrics: Dict) -> Dict:
        """Generate targeting approach for a segment"""
        try:
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
        except Exception as e:
            logger.error(f"Error generating segment approach: {str(e)}")
            raise

    def _identify_suitable_products(self, metrics: Dict) -> List[Dict]:
        """Identify suitable loan products for segment"""
        try:
            products = []
            avg_amount = metrics.get('avg_loan_amount', 0)

            if avg_amount > 100000:
                products.append({
                    'type': 'Business Loan',
                    'target_amount': float(avg_amount),
                    'success_probability': float(metrics.get('approval_rate', 0))
                })
            if 30000 <= avg_amount <= 100000:
                products.append({
                    'type': 'Personal Loan',
                    'target_amount': float(avg_amount),
                    'success_probability': float(metrics.get('approval_rate', 0))
                })
            return products
        except Exception as e:
            logger.error(f"Error identifying suitable products: {str(e)}")
            return []

    def _identify_growth_opportunities(self, analysis: Dict) -> List[Dict]:
        """Identify growth opportunities in the market"""
        try:
            growth_patterns = analysis['growth_patterns']
            opportunities = []

            for loan_type, metrics in growth_patterns['loan_type_growth'].items():
                if metrics['share_change'] > 0 and metrics['approval_rate'] > 0.6:
                    opportunities.append({
                        'product': loan_type,
                        'growth_rate': float(metrics['share_change']),
                        'success_rate': float(metrics['approval_rate']),
                        'recommendation': self._generate_growth_recommendation(metrics)
                    })

            return sorted(opportunities, key=lambda x: x['growth_rate'], reverse=True)
        except Exception as e:
            logger.error(f"Error identifying growth opportunities: {str(e)}")
            return []

    def _generate_growth_recommendation(self, metrics: Dict) -> Dict:
        """Generate specific growth recommendations"""
        try:
            return {
                'strategy': 'Expansion' if metrics['share_change'] > 0.1 else 'Optimization',
                'focus_areas': [
                    'Market penetration' if metrics['approval_rate'] > 0.7 else 'Risk optimization',
                    'Volume growth' if metrics['share_change'] > 0.15 else 'Quality focus'
                ]
            }
        except Exception as e:
            logger.error(f"Error generating growth recommendation: {str(e)}")
            return {'strategy': 'Optimization', 'focus_areas': ['Risk optimization']}

    def _generate_risk_mitigation_strategies(self, analysis: Dict) -> List[Dict]:
        """Generate risk mitigation strategies"""
        try:
            risk_assessment = analysis['risk_assessment']
            strategies = []

            # Handle concentration risk
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

            # Handle volatility risk
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

            # Handle rejection risk
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
        except Exception as e:
            logger.error(f"Error generating risk mitigation strategies: {str(e)}")
            return []