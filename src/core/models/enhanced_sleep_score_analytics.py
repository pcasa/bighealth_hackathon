# Integration with the existing SleepScoreAnalytics class

import os
from src.core.models.sleep_score_analytics import SleepScoreAnalytics
from src.core.models.trend_analysis_extension import TrendAnalysisExtension

# Create extension class and inherit from SleepScoreAnalytics
class EnhancedSleepScoreAnalytics(SleepScoreAnalytics):
    """Enhanced version of SleepScoreAnalytics with trend analysis capabilities"""
    
    def __init__(self, sleep_quality_model=None):
        """Initialize the enhanced analytics module"""
        super().__init__(sleep_quality_model)
        self.trend_extension = TrendAnalysisExtension()
    
    def analyze_trends(self, data, metric='sleep_score', min_periods=3):
        """Analyze trends across all dimensions"""
        # Set full dataset in the trend extension
        self.trend_extension.full_data_set = self.full_data_set if hasattr(self, 'full_data_set') else data
        
        # Run trend analysis
        return self.trend_extension.analyze_all_dimension_trends(data, metric, min_periods)
    
    def create_trend_visualizations(self, trend_results, output_dir):
        """Create visualizations for trend analysis"""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations
        return self.trend_extension.visualize_dimension_trends(trend_results, output_dir)
    
    def generate_trend_report(self, trend_results, output_path):
        """Generate a trend analysis report"""
        return self.trend_extension.generate_trend_report(trend_results, output_path)
    
    def run_full_analysis(self, data, output_dir='reports'):
        """Run both static analysis and trend analysis"""
        # First ensure we have the full dataset available
        self.set_full_dataset(data)
        
        # Calculate sleep scores if not already done
        if 'sleep_score' not in data.columns:
            data = self.calculate_sleep_scores(data)
        
        # Ensure output directories exist
        static_output_dir = os.path.join(output_dir, 'static_analysis')
        trend_output_dir = os.path.join(output_dir, 'trend_analysis')
        os.makedirs(static_output_dir, exist_ok=True)
        os.makedirs(trend_output_dir, exist_ok=True)
        
        # Run static analysis (existing functionality)
        static_results = self.analyze_all_dimensions(data)
        self.create_visualizations(static_results, static_output_dir)
        self.generate_summary_report(static_results, os.path.join(static_output_dir, 'summary_report.md'))
        
        # Run trend analysis (new functionality)
        trend_results = self.analyze_trends(data)
        self.create_trend_visualizations(trend_results, trend_output_dir)
        self.generate_trend_report(trend_results, os.path.join(trend_output_dir, 'trend_report.md'))
        
        return {
            'static_results': static_results,
            'trend_results': trend_results
        }