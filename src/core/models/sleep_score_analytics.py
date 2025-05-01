#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for analyzing sleep scores across demographic and temporal dimensions.
This module works with the existing Sleep Insights App architecture.
"""
from pydantic import Field, validator
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from core.models.base_model import BaseModel
from src.core.models.sleep_quality import SleepQualityModel
from src.core.data_processing.preprocessing import Preprocessor
from src.utils.constants import profession_categories
from src.core.models.improved_sleep_score import ImprovedSleepScoreCalculator
from src.core.scoring.sleep_score import SleepScoreCalculator

class AnalysisDimension(str, Enum):
    """Dimensions for sleep score analysis"""
    AGE_RANGE = "age_range"
    PROFESSION = "profession_category"
    REGION = "region_category"
    SEASON = "season"
    GENDER = "gender"
    WEEKEND = "is_weekend"


class AnalysisMetric(str, Enum):
    """Metrics for sleep score analysis"""
    SLEEP_SCORE = "sleep_score"
    SLEEP_EFFICIENCY = "sleep_efficiency"
    SLEEP_DURATION = "sleep_duration_hours"
    SUBJECTIVE_RATING = "subjective_rating"


class AnalysisStats(BaseModel):
    """Statistical analysis for a category"""
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float


class DimensionAnalysis(BaseModel):
    """Analysis results for a specific dimension"""
    dimension: AnalysisDimension
    metric: AnalysisMetric
    results: Dict[str, AnalysisStats]
    visualization_path: Optional[str] = None


class CrossDimensionAnalysis(BaseModel):
    """Analysis results across two dimensions"""
    primary_dimension: AnalysisDimension
    secondary_dimension: AnalysisDimension
    metric: AnalysisMetric
    heatmap_data: Dict[str, Dict[str, float]]
    visualization_path: Optional[str] = None


class SleepScoreAnalytics:
    """Class for analyzing sleep scores across different dimensions"""
    
    def __init__(self, sleep_quality_model=None):
        """Initialize the analytics module"""
        # Use provided model or create a new one
        self.sleep_quality_model = sleep_quality_model or SleepQualityModel()
        self.preprocessor = Preprocessor()
        self.sleep_score_calculator = ImprovedSleepScoreCalculator()
        
        # Define age ranges for analysis
        self.age_ranges = [
            (18, 29, "18-29"),
            (30, 39, "30-39"),
            (40, 49, "40-49"),
            (50, 59, "50-59"),
            (60, 69, "60-69"),
            (70, 85, "70+")
        ]
        
        # Define seasons (Northern Hemisphere)
        self.seasons = {
            (12, 1, 2): "Winter",
            (3, 4, 5): "Spring",
            (6, 7, 8): "Summer",
            (9, 10, 11): "Fall"
        }
    
    def load_data(self, sleep_data_path, users_data_path, wearable_data_path=None):
        """Load and preprocess data for analysis"""
        # Load data
        sleep_data = pd.read_csv(sleep_data_path)
        users_data = pd.read_csv(users_data_path)
        
        if wearable_data_path:
            wearable_data = pd.read_csv(wearable_data_path)
        else:
            wearable_data = None
        
        # Merge sleep data with user profiles
        merged_data = pd.merge(sleep_data, users_data, on='user_id')
        
        # Preprocess data
        processed_data = self.preprocessor.preprocess_sleep_data(merged_data, wearable_data)
        
        # Ensure date is in datetime format
        processed_data['date'] = pd.to_datetime(processed_data['date'])
        
        # Add season column
        processed_data['month'] = processed_data['date'].dt.month
        processed_data['season'] = processed_data['month'].apply(self._get_season)
        
        # Add age range column
        processed_data['age_range'] = processed_data['age'].apply(self._get_age_range)
        
        # Categorize professions if not already done
        if 'profession_category' not in processed_data.columns:
            processed_data['profession_category'] = processed_data['profession'].apply(self._categorize_profession)
        
        # Categorize regions if not already done
        if 'region_category' not in processed_data.columns:
            processed_data['region_category'] = processed_data['region'].apply(self._categorize_region)
        
        return processed_data
    
    def calculate_sleep_scores(self, data):
        """Calculate sleep scores for each record"""
        scored_data = data.copy()
        
        # Add sleep score column using the centralized calculator
        scored_data['sleep_score'] = scored_data.apply(
            lambda row: self.sleep_score_calculator.calculate_score(row.to_dict()).total_score,
            axis=1
        )
        
        return scored_data
    
    def analyze_by_dimension(self, data, dimension, metric='sleep_score') -> DimensionAnalysis:
        """Analyze metrics by a specific dimension"""
        # Validate dimension and metric
        try:
            dim = AnalysisDimension(dimension)
            met = AnalysisMetric(metric)
        except ValueError:
            # Fallback for custom dimensions or metrics
            dim = dimension
            met = metric
        
        # Group by dimension and calculate stats
        grouped = data.groupby(dimension)[metric].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).reset_index()
        
        # Sort by mean score descending
        sorted_data = grouped.sort_values('mean', ascending=False)
        
        # Convert to DimensionAnalysis model
        results = {}
        for _, row in sorted_data.iterrows():
            category = row[dimension]
            results[category] = AnalysisStats(
                count=int(row['count']),
                mean=float(row['mean']),
                median=float(row['median']),
                std=float(row['std']),
                min=float(row['min']),
                max=float(row['max'])
            )
        
        return DimensionAnalysis(
            dimension=dim,
            metric=met,
            results=results
        )
    
    def analyze_all_dimensions(self, data, metric='sleep_score'):
        """Analyze metrics across all key dimensions"""
        results = {}
        
        # Analyze each dimension
        dimensions = ['age_range', 'profession_category', 'region_category', 'season']
        for dimension in dimensions:
            results[dimension] = self.analyze_by_dimension(data, dimension, metric)
        
        # Additional analysis by gender
        if 'gender' in data.columns:
            results['gender'] = self.analyze_by_dimension(data, 'gender', metric)
        
        # Analysis by weekday/weekend
        if 'is_weekend' in data.columns:
            # Store the results with key 'is_weekend' instead of 'weekday_weekend'
            results['is_weekend'] = self.analyze_by_dimension(data, 'is_weekend', metric)
        
        return results
    
    def create_visualizations(self, analysis_results, output_dir):
        """Create visualizations for each dimension"""
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")
        
        # Create visualizations for each dimension
        for dimension, data in analysis_results.items():
            self._create_dimension_plot(dimension, data, output_dir)
        
        # Create cross-dimension heatmaps
        if 'profession_category' in analysis_results and 'region_category' in analysis_results:
            self._create_cross_dimension_heatmap(
                'profession_category', 
                'region_category',
                analysis_results,
                output_dir
            )
        
        if 'age_range' in analysis_results and 'season' in analysis_results:
            self._create_cross_dimension_heatmap(
                'age_range', 
                'season',
                analysis_results,
                output_dir
            )
    
    def generate_summary_report(self, analysis_results, output_path):
        """Generate a summary report of the analysis"""
        with open(output_path, 'w') as f:
            f.write("# Sleep Score Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary for each dimension
            for dimension, data in analysis_results.items():
                f.write(f"## Sleep Scores by {dimension.replace('_', ' ').title()}\n\n")
                
                # Markdown table header
                f.write("| Category | Count | Mean Score | Median | Std Dev | Min | Max |\n")
                f.write("|----------|-------|------------|--------|---------|-----|-----|\n")
                
                # Write each row
                for _, row in data.iterrows():
                    f.write(f"| {row[dimension]} | {int(row['count'])} | {row['mean']:.2f} | ")
                    f.write(f"{row['median']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['max']:.2f} |\n")
                
                f.write("\n")
                
                # Add insights
                f.write("### Key Insights\n\n")
                
                best_category = data.iloc[0][dimension]
                worst_category = data.iloc[-1][dimension]
                mean_diff = data.iloc[0]['mean'] - data.iloc[-1]['mean']
                
                f.write(f"- {best_category} has the highest average sleep score ({data.iloc[0]['mean']:.2f}).\n")
                f.write(f"- {worst_category} has the lowest average sleep score ({data.iloc[-1]['mean']:.2f}).\n")
                f.write(f"- The difference between the highest and lowest category is {mean_diff:.2f} points.\n")
                
                # Identify highest variability
                highest_std_idx = data['std'].idxmax()
                highest_std_cat = data.iloc[highest_std_idx][dimension]
                highest_std = data.iloc[highest_std_idx]['std']
                
                f.write(f"- {highest_std_cat} shows the highest variability with a standard deviation of {highest_std:.2f}.\n\n")
                
                # Add image reference
                f.write(f"![Sleep Scores by {dimension.replace('_', ' ').title()}]({dimension}_sleep_scores.png)\n\n")
            
            # Add cross-dimension insights
            f.write("## Cross-Dimensional Insights\n\n")
            
            if 'profession_category' in analysis_results and 'region_category' in analysis_results:
                f.write("### Profession and Region Interaction\n\n")
                f.write("The heatmap below shows how sleep scores vary across both profession and region categories:\n\n")
                f.write("![Profession-Region Heatmap](profession_region_heatmap.png)\n\n")
            
            if 'age_range' in analysis_results and 'season' in analysis_results:
                f.write("### Age Range and Season Interaction\n\n")
                f.write("The heatmap below shows how sleep scores vary across both age ranges and seasons:\n\n")
                f.write("![Age-Season Heatmap](age_season_heatmap.png)\n\n")
            
            # Final recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the analysis, we recommend:\n\n")
            
            # Generate dynamic recommendations based on findings
            recommendations = self._generate_recommendations(analysis_results)
            for rec in recommendations:
                f.write(f"- {rec}\n")
        
        print(f"Summary report saved to {output_path}")
    
    def _get_age_range(self, age):
        """Get the age range label for a given age"""
        for min_age, max_age, label in self.age_ranges:
            if min_age <= age <= max_age:
                return label
        return "Unknown"
    
    def _get_season(self, month):
        """Get the season for a given month"""
        for months, season in self.seasons.items():
            if month in months:
                return season
        return "Unknown"
    
    def _categorize_profession(self, profession):
        """Categorize profession based on keywords"""
        for category, keywords in profession_categories.items():
            if any(keyword.lower() in profession.lower() for keyword in keywords):
                return category
        return "other"
    
    def _categorize_region(self, region):
        """Categorize region based on country"""
        if not isinstance(region, str) or ',' not in region:
            return "other"
            
        parts = region.split(',')
        country = parts[-1].strip()
        
        north_america = ['United States', 'Canada', 'Mexico', 'USA']
        europe = ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'UK']
        asia = ['China', 'Japan', 'India', 'Korea', 'Thailand', 'Singapore']
        
        if country in north_america:
            return "north_america"
        elif country in europe:
            return "europe"
        elif country in asia:
            return "asia"
        else:
            return "other"
    
    def _calculate_score_for_row(self, row):
        """Calculate sleep score for a single row"""
        # Convert row to dictionary (if it's a pandas Series)
        sleep_data = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
        
        # Calculate score using the improved calculator
        return self.sleep_score_calculator.calculate_score(sleep_data)
    
        """Calculate sleep score for a single row"""
        # Extract basic metrics
        sleep_efficiency = row.get('sleep_efficiency', 0)
        subjective_rating = row.get('subjective_rating')
        sleep_data = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
        
        # Create additional metrics dictionary
        additional_metrics = {}
        for col in ['deep_sleep_percentage', 'rem_sleep_percentage', 
                   'sleep_onset_latency_minutes', 'awakenings_count']:
            if col in row:
                additional_metrics[col] = row[col]
        
        # Calculate score
        try:
            return self.sleep_quality_model.calculate_sleep_score(
                sleep_efficiency, 
                subjective_rating,
                additional_metrics
            )
        except Exception as e:
            # Fallback if model fails
            print(f"Warning: Model-based score calculation failed: {str(e)}")
            # Simple calculation
            score = int(sleep_efficiency * 80)
            if subjective_rating is not None:
                score += int(subjective_rating * 2)
            return min(100, max(0, score))
    
    def _create_dimension_plot(self, dimension, data, output_dir):
        """Create and save plot for a specific dimension"""
        plt.figure(figsize=(12, 6))
        
        # Bar plot with error bars
        bars = plt.bar(data[dimension], data['mean'], yerr=data['std'], capsize=5)
        
        # Customize plot
        plt.xlabel(dimension.replace('_', ' ').title())
        plt.ylabel('Average Sleep Score')
        plt.title(f'Sleep Scores by {dimension.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Save figure
        plt.savefig(f"{output_dir}/{dimension}_sleep_scores.png", dpi=300)
        plt.close()
    
    def _create_cross_dimension_heatmap(self, dim1, dim2, analysis_results, output_dir):
        """Create heatmap showing interaction between two dimensions"""
        # This requires the original data, not the analysis results
        # So we'll create a helper method to process the pivot table
        
        # Check if we have the full data set
        if not hasattr(self, 'full_data_set'):
            print(f"Warning: Full dataset not available for cross-dimension heatmap.")
            return
        
        # Create the cross-tabulation
        pivot_table = pd.pivot_table(
            self.full_data_set, 
            values='sleep_score', 
            index=dim1, 
            columns=dim2,
            aggfunc='mean'
        )
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", cbar_kws={'label': 'Average Sleep Score'})
        plt.title(f'Sleep Score by {dim1.replace("_", " ").title()} and {dim2.replace("_", " ").title()}')
        plt.tight_layout()
        
        # Save figure
        filename = f"{dim1}_{dim2}_heatmap.png"
        plt.savefig(f"{output_dir}/{filename}", dpi=300)
        plt.close()
    
    def _generate_recommendations(self, analysis_results):
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Profession-based recommendations
        if 'profession_category' in analysis_results:
            prof_data = analysis_results['profession_category']
            worst_prof = prof_data.iloc[-1]['profession_category']
            
            if worst_prof == 'healthcare':
                recommendations.append("Implement targeted sleep improvement programs for healthcare professionals, focusing on strategies for shift work and stress management.")
            elif worst_prof == 'tech':
                recommendations.append("Develop guidelines for tech workers on reducing blue light exposure and creating better work-life boundaries.")
        
        # Age-based recommendations
        if 'age_range' in analysis_results:
            age_data = analysis_results['age_range']
            worst_age = age_data.iloc[-1]['age_range']
            
            if worst_age in ['18-29', '30-39']:
                recommendations.append("Create sleep education targeted at younger adults emphasizing the importance of consistent sleep schedules.")
            elif worst_age in ['60-69', '70+']:
                recommendations.append("Develop specialized sleep hygiene recommendations for older adults addressing age-specific sleep challenges.")
        
        # Region-based recommendations
        if 'region_category' in analysis_results:
            region_data = analysis_results['region_category']
            worst_region = region_data.iloc[-1]['region_category']
            
            if worst_region == 'north_america':
                recommendations.append("Develop culturally appropriate interventions addressing work-life balance for North American users.")
            elif worst_region == 'asia':
                recommendations.append("Create recommendations for Asian users addressing regional factors like urban density and light pollution.")
        
        # General recommendations
        recommendations.append("Continue collecting demographic data to refine understanding of sleep quality determinants across different populations.")
        recommendations.append("Develop personalized recommendation algorithms that factor in age, profession, region, and season.")
        
        return recommendations

    def set_full_dataset(self, data):
        """Set the full dataset for cross-dimensional analysis"""
        self.full_data_set = data

    def analyze_season_impact(self, data):
        """Analyze seasonal impact on sleep metrics"""
        # Group by season and calculate sleep metrics
        seasonal_impact = data.groupby('season').agg({
            'sleep_score': ['mean', 'std'],
            'sleep_efficiency': ['mean', 'std'],
            'sleep_duration_hours': ['mean', 'std'],
            'sleep_onset_latency_minutes': ['mean', 'std'],
            'awakenings_count': ['mean', 'std'],
            'subjective_rating': ['mean', 'std']
        })
        
        return seasonal_impact


# Example usage:
if __name__ == "__main__":
    # Initialize the analytics module
    analytics = SleepScoreAnalytics()
    
    # Load and process data
    data = analytics.load_data(
        sleep_data_path='data/raw/sleep_data.csv',
        users_data_path='data/raw/users.csv',
        wearable_data_path='data/raw/wearable_data.csv'
    )
    
    # Calculate sleep scores
    scored_data = analytics.calculate_sleep_scores(data)
    
    # Set full dataset for cross-dimensional analysis
    analytics.set_full_dataset(scored_data)
    
    # Analyze across all dimensions
    results = analytics.analyze_all_dimensions(scored_data)
    
    # Create visualizations
    analytics.create_visualizations(results, 'reports/sleep_score_analysis')
    
    # Generate summary report
    analytics.generate_summary_report(results, 'reports/sleep_score_analysis/summary_report.md')
    
    print("Analysis complete. Results saved to reports/sleep_score_analysis/")