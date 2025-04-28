#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extension to the SleepScoreAnalytics class that adds trend analysis 
across demographic and temporal dimensions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

class TrendAnalysisExtension:
    """Extension class to add trend analysis to the SleepScoreAnalytics class"""
    
    def analyze_trends_by_dimension(self, data, dimension, metric='sleep_score', min_periods=3):
        """
        Analyze improvement/worsening trends by demographic dimension.
        
        Args:
            data: DataFrame containing sleep data with dates and dimensions
            dimension: The demographic dimension to analyze (age_range, profession_category, etc.)
            metric: The metric to track trends for (sleep_score, sleep_efficiency, etc.)
            min_periods: Minimum number of time periods needed to calculate a trend
            
        Returns:
            DataFrame with trend analysis results sorted by trend value (descending)
        """
        # Ensure date is in datetime format
        if not pd.api.types.is_datetime64_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
        
        # Create week number for time bucketing
        data['week'] = data['date'].dt.isocalendar().week
        
        # Initialize results containers
        trends = []
        
        # Process each category in the dimension
        for category in data[dimension].unique():
            # Skip missing/nan values
            if pd.isna(category):
                continue
                
            # Get data for this category
            category_data = data[data[dimension] == category]
            
            # Calculate weekly averages
            weekly_averages = category_data.groupby('week')[metric].mean()
            
            # Only calculate trend if we have enough time periods
            if len(weekly_averages) >= min_periods:
                # Calculate linear trend (slope)
                x = np.arange(len(weekly_averages))
                trend_slope, intercept = np.polyfit(x, weekly_averages.values, 1)
                
                # Calculate trend quality metrics
                r_squared = self._calculate_r_squared(x, weekly_averages.values, trend_slope, intercept)
                
                # Calculate start and end values
                start_value = weekly_averages.iloc[0]
                end_value = weekly_averages.iloc[-1]
                pct_change = ((end_value - start_value) / start_value) * 100 if start_value > 0 else 0
                
                # Calculate mean value
                mean_value = weekly_averages.mean()
                
                # Add to results
                trends.append({
                    dimension: category,
                    'trend_slope': trend_slope,
                    'r_squared': r_squared,
                    'start_value': start_value,
                    'end_value': end_value,
                    'mean_value': mean_value,
                    'percent_change': pct_change,
                    'num_weeks': len(weekly_averages),
                    'trend_direction': 'Improving' if trend_slope > 0 else 'Worsening' if trend_slope < 0 else 'Stable'
                })
        
        # Convert to DataFrame
        trends_df = pd.DataFrame(trends)
        
        # Sort by trend slope (descending)
        if not trends_df.empty:
            trends_df = trends_df.sort_values('trend_slope', ascending=False)
        
        return trends_df
    
    def _calculate_r_squared(self, x, y, slope, intercept):
        """Calculate R-squared value for the trend line"""
        y_pred = slope * x + intercept
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        
        # Avoid division by zero
        if ss_total == 0:
            return 0
            
        return 1 - (ss_residual / ss_total)
    
    def analyze_all_dimension_trends(self, data, metric='sleep_score', min_periods=3):
        """
        Analyze trends across all demographic dimensions
        
        Args:
            data: DataFrame containing sleep data
            metric: The metric to track trends for
            min_periods: Minimum number of time periods needed to calculate a trend
            
        Returns:
            Dictionary with trend results for each dimension
        """
        results = {}
        
        # Analyze each dimension
        dimensions = ['age_range', 'profession_category', 'region_category', 'season']
        for dimension in dimensions:
            if dimension in data.columns:
                results[dimension] = self.analyze_trends_by_dimension(
                    data, dimension, metric, min_periods
                )
        
        # Additional dimensions if available
        if 'gender' in data.columns:
            results['gender'] = self.analyze_trends_by_dimension(
                data, 'gender', metric, min_periods
            )
        
        if 'is_weekend' in data.columns:
            results['is_weekend'] = self.analyze_trends_by_dimension(
                data, 'is_weekend', metric, min_periods
            )
        
        return results
    
    def visualize_dimension_trends(self, trend_results, output_dir):
        """
        Create visualizations for trend analysis results
        
        Args:
            trend_results: Dictionary with trend results from analyze_all_dimension_trends
            output_dir: Directory to save visualization files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize each dimension's trends
        for dimension, data in trend_results.items():
            if data is None or len(data) == 0:
                continue
                
            self._create_trend_plot(dimension, data, output_dir)
            self._create_weekly_progression_plot(dimension, data, output_dir)
    
    def _create_trend_plot(self, dimension, data, output_dir):
        """Create bar plot of trend slopes by category"""
        if 'trend_slope' not in data.columns or len(data) == 0:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Create color map based on trend direction
        colors = ['green' if slope > 0 else 'red' if slope < 0 else 'gray' 
                 for slope in data['trend_slope']]
        
        # Bar plot of trend slopes
        bars = plt.bar(data[dimension], data['trend_slope'], color=colors)
        
        # Add reference line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Customize plot
        plt.xlabel(dimension.replace('_', ' ').title())
        plt.ylabel(f'Trend Slope (Change per Week)')
        plt.title(f'Sleep Improvement Trends by {dimension.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                va = 'bottom'
                offset = 0.01
            else:
                va = 'top'
                offset = -0.01
                
            plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{height:.2f}', ha='center', va=va)
        
        # Save figure
        plt.savefig(f"{output_dir}/{dimension}_trends.png", dpi=300)
        plt.close()
    
    def _create_weekly_progression_plot(self, dimension, trend_data, output_dir, max_categories=5):
        """
        Create line plot showing weekly progression for each category
        
        Args:
            dimension: The demographic dimension being analyzed
            trend_data: DataFrame with trend analysis for this dimension
            output_dir: Directory to save visualization
            max_categories: Maximum number of categories to include in the plot
        """
        # Skip if trend_data is empty
        if len(trend_data) == 0:
            return
        
        # We need to regenerate the weekly data from the original dataset
        if not hasattr(self, 'full_data_set'):
            print(f"Warning: Full dataset not available for weekly progression plot.")
            return
        
        # Sort categories by trend_slope
        sorted_categories = trend_data.sort_values('trend_slope', ascending=False)[dimension].tolist()
        
        # Limit to top and bottom categories
        if len(sorted_categories) > max_categories:
            top_categories = sorted_categories[:max_categories//2]
            bottom_categories = sorted_categories[-max_categories//2:]
            categories_to_plot = top_categories + bottom_categories
        else:
            categories_to_plot = sorted_categories
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Convert to datetime if needed
        data = self.full_data_set.copy()
        if not pd.api.types.is_datetime64_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
            
        # Create week number
        data['week'] = data['date'].dt.isocalendar().week
        
        # Plot each category
        for category in categories_to_plot:
            category_data = data[data[dimension] == category]
            
            # Group by week and calculate average
            weekly_avg = category_data.groupby('week')['sleep_score'].mean()
            
            # Get trend slope for this category
            trend_slope = trend_data.loc[trend_data[dimension] == category, 'trend_slope'].values[0]
            
            # Format label with trend direction
            direction = "↗" if trend_slope > 0 else "↘" if trend_slope < 0 else "→"
            label = f"{category} ({direction} {trend_slope:.2f})"
            
            # Plot weekly progression
            plt.plot(weekly_avg.index, weekly_avg.values, 'o-', linewidth=2, markersize=6, label=label)
            
        # Customize plot
        plt.xlabel('Week')
        plt.ylabel('Average Sleep Score')
        plt.title(f'Weekly Sleep Score Progression by {dimension.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{output_dir}/{dimension}_weekly_progression.png", dpi=300)
        plt.close()
    
    def generate_trend_report(self, trend_results, output_path):
        """
        Generate a comprehensive report of trend analysis results
        
        Args:
            trend_results: Dictionary with trend results from analyze_all_dimension_trends
            output_path: File path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("# Sleep Score Trend Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("This report analyzes improvement and worsening trends in sleep scores across different demographic dimensions.\n\n")
            
            # Write summary for each dimension
            for dimension, data in trend_results.items():
                if data is None or len(data) == 0:
                    continue
                    
                f.write(f"## Trends by {dimension.replace('_', ' ').title()}\n\n")
                
                # Markdown table header
                f.write("| Category | Trend Direction | Trend Slope | Start | End | % Change | Weeks |\n")
                f.write("|----------|----------------|-------------|-------|-----|----------|-------|\n")
                
                # Write each row
                for _, row in data.iterrows():
                    direction = "⬆️" if row['trend_slope'] > 0 else "⬇️" if row['trend_slope'] < 0 else "➡️"
                    
                    f.write(f"| {row[dimension]} | {direction} {row['trend_direction']} | {row['trend_slope']:.3f} | ")
                    f.write(f"{row['start_value']:.1f} | {row['end_value']:.1f} | {row['percent_change']:.1f}% | {row['num_weeks']} |\n")
                
                f.write("\n")
                
                # Add insights
                f.write("### Key Insights\n\n")
                
                # Find fastest improving and worsening categories
                if len(data) > 1:
                    best_category = data.iloc[0][dimension]
                    best_slope = data.iloc[0]['trend_slope']
                    worst_category = data.iloc[-1][dimension]
                    worst_slope = data.iloc[-1]['trend_slope']
                    
                    # Only mention if there are actual trends
                    if best_slope > 0:
                        f.write(f"- **{best_category}** shows the fastest improvement with a slope of {best_slope:.3f} points per week.\n")
                    
                    if worst_slope < 0:
                        f.write(f"- **{worst_category}** shows the fastest decline with a slope of {worst_slope:.3f} points per week.\n")
                
                # Add image references
                f.write(f"\n![Trends by {dimension.replace('_', ' ').title()}]({dimension}_trends.png)\n\n")
                f.write(f"![Weekly Progression by {dimension.replace('_', ' ').title()}]({dimension}_weekly_progression.png)\n\n")
            
            # Add cross-dimensional insights
            f.write("## Cross-Dimensional Trend Insights\n\n")
            
            # Generate insights by looking at trends across dimensions
            cross_insights = self._generate_cross_dimension_insights(trend_results)
            for insight in cross_insights:
                f.write(f"- {insight}\n")
            
            f.write("\n")
            
            # Recommendations based on trends
            f.write("## Recommendations Based on Trends\n\n")
            recommendations = self._generate_trend_based_recommendations(trend_results)
            for rec in recommendations:
                f.write(f"- {rec}\n")
        
        print(f"Trend report saved to {output_path}")
    
    def _generate_cross_dimension_insights(self, trend_results):
        """Generate insights by analyzing trends across different dimensions"""
        insights = []
        
        # Compare trends across age groups
        if 'age_range' in trend_results and len(trend_results['age_range']) > 0:
            age_trends = trend_results['age_range']
            young_trends = age_trends[age_trends['age_range'].isin(['18-29', '30-39'])]
            old_trends = age_trends[age_trends['age_range'].isin(['60-69', '70+'])]
            
            if not young_trends.empty and not old_trends.empty:
                young_avg = young_trends['trend_slope'].mean()
                old_avg = old_trends['trend_slope'].mean()
                
                if young_avg > old_avg:
                    insights.append(f"Younger age groups (18-39) show faster improvement ({young_avg:.3f}) than older age groups (60+) ({old_avg:.3f}).")
                elif old_avg > young_avg:
                    insights.append(f"Older age groups (60+) show faster improvement ({old_avg:.3f}) than younger age groups (18-39) ({young_avg:.3f}).")
        
        # Compare trends across professions
        if 'profession_category' in trend_results and len(trend_results['profession_category']) > 0:
            prof_trends = trend_results['profession_category']
            
            # Find professions with positive vs negative trends
            improving_profs = prof_trends[prof_trends['trend_slope'] > 0]
            worsening_profs = prof_trends[prof_trends['trend_slope'] < 0]
            
            if len(improving_profs) > 0:
                top_prof = improving_profs.iloc[0]['profession_category'] 
                insights.append(f"The {top_prof} profession shows the strongest positive trend, suggesting effective sleep adaptation.")
            
            if len(worsening_profs) > 0:
                bottom_prof = worsening_profs.iloc[-1]['profession_category']
                insights.append(f"The {bottom_prof} profession shows the strongest negative trend, suggesting growing sleep challenges.")
        
        # Compare trends across regions
        if 'region_category' in trend_results and len(trend_results['region_category']) > 0:
            region_trends = trend_results['region_category']
            
            if len(region_trends) > 1:
                regions_sorted = region_trends.sort_values('trend_slope', ascending=False)
                best_region = regions_sorted.iloc[0]['region_category']
                worst_region = regions_sorted.iloc[-1]['region_category']
                
                insights.append(f"The {best_region} region shows the most positive sleep trend while {worst_region} shows the least favorable trend.")
        
        # Analyze seasonal trends
        if 'season' in trend_results and len(trend_results['season']) > 0:
            season_trends = trend_results['season']
            
            if len(season_trends) > 1:
                best_season = season_trends.iloc[0]['season']
                insights.append(f"Sleep improvement trends are strongest during {best_season}, suggesting seasonal factors impact sleep adaptation.")
        
        return insights
    
    def _generate_trend_based_recommendations(self, trend_results):
        """Generate recommendations based on trend analysis"""
        recommendations = []
        
        # Age-based recommendations
        if 'age_range' in trend_results and len(trend_results['age_range']) > 0:
            age_trends = trend_results['age_range']
            
            # Find age groups with negative trends
            worsening_ages = age_trends[age_trends['trend_slope'] < 0]
            
            if len(worsening_ages) > 0:
                worst_age = worsening_ages.iloc[-1]['age_range']
                recommendations.append(f"Develop targeted interventions for the {worst_age} age group, which shows the strongest negative sleep trend.")
        
        # Profession-based recommendations
        if 'profession_category' in trend_results and len(trend_results['profession_category']) > 0:
            prof_trends = trend_results['profession_category']
            worsening_profs = prof_trends[prof_trends['trend_slope'] < 0]
            
            if len(worsening_profs) > 0:
                worst_prof = worsening_profs.iloc[-1]['profession_category']
                
                if worst_prof == 'healthcare':
                    recommendations.append("Implement specialized support for healthcare professionals, focusing on shift adjustment techniques and stress management.")
                elif worst_prof == 'tech':
                    recommendations.append("Develop specific guidelines for tech professionals focusing on screen time management and work-life boundaries.")
                elif worst_prof == 'service':
                    recommendations.append("Create resources for service industry workers to manage irregular schedules and develop consistent sleep routines.")
                else:
                    recommendations.append(f"Develop targeted sleep improvement strategies for the {worst_prof} profession to address their worsening sleep trends.")
        
        # Region-based recommendations
        if 'region_category' in trend_results and len(trend_results['region_category']) > 0:
            region_trends = trend_results['region_category']
            worsening_regions = region_trends[region_trends['trend_slope'] < 0]
            
            if len(worsening_regions) > 0:
                worst_region = worsening_regions.iloc[-1]['region_category']
                recommendations.append(f"Develop culturally appropriate sleep interventions for users in the {worst_region} region to address declining sleep trends.")
        
        # Season-based recommendations
        if 'season' in trend_results and len(trend_results['season']) > 0:
            season_trends = trend_results['season']
            
            if len(season_trends) > 1:
                worst_season = season_trends.iloc[-1]['season']
                recommendations.append(f"Create season-specific recommendations for {worst_season}, when sleep trends are least favorable.")
        
        # General recommendations
        recommendations.append("Implement continuous trend monitoring by demographic factors to identify emerging sleep challenges.")
        recommendations.append("Create a targeted early intervention system for user groups showing negative sleep trends.")
        
        return recommendations