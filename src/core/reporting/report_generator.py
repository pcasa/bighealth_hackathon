"""
Module for generating user sleep reports in various formats.
"""

import os
from datetime import datetime


def create_user_report(user_profile, sleep_metrics, recommendations, visualization_dir, output_dir):
    """
    Create a comprehensive HTML report for a user.
    
    Args:
        user_profile: Dictionary containing user profile information
        sleep_metrics: Dictionary of calculated sleep metrics
        recommendations: List of recommendation dictionaries
        visualization_dir: Directory containing visualizations
        output_dir: Directory to save the report
        
    Returns:
        str: Path to the generated report
    """
    user_id = user_profile['user_id']
    report_path = os.path.join(output_dir, f"{user_id}_sleep_insights.html")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Format metrics for display
    formatted_metrics = _format_metrics_for_display(sleep_metrics)
    
    # Generate relative paths for visualizations
    visualization_paths = _get_visualization_paths(user_id)
    
    # Generate HTML content
    html_content = _generate_html_content(
        user_profile,
        formatted_metrics,
        recommendations,
        visualization_paths
    )
    
    # Write to file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path


def _format_metrics_for_display(sleep_metrics):
    """Format metrics for display in the report."""
    formatted_metrics = {
        'avg_sleep_efficiency': f"{sleep_metrics['avg_sleep_efficiency']*100:.1f}%",
        'avg_sleep_duration': f"{sleep_metrics['avg_sleep_duration']:.1f} hours",
        'avg_time_in_bed': f"{sleep_metrics['avg_time_in_bed']:.1f} hours",
        'avg_sleep_onset': f"{sleep_metrics['avg_sleep_onset']:.1f} minutes",
        'avg_awakenings': f"{sleep_metrics['avg_awakenings']:.1f}",
        'avg_sleep_rating': f"{sleep_metrics['avg_sleep_rating']:.1f}/10"
    }
    
    # Add trend metrics if available
    if 'efficiency_change' in sleep_metrics:
        efficiency_trend = "↑" if sleep_metrics['efficiency_change'] > 0 else "↓" if sleep_metrics['efficiency_change'] < 0 else "→"
        formatted_metrics['efficiency_trend'] = f"{efficiency_trend} {abs(sleep_metrics['efficiency_change']*100):.1f}%"
        
        duration_trend = "↑" if sleep_metrics['duration_change'] > 0 else "↓" if sleep_metrics['duration_change'] < 0 else "→"
        formatted_metrics['duration_trend'] = f"{duration_trend} {abs(sleep_metrics['duration_change']):.1f} hours"
    
    return formatted_metrics


def _get_visualization_paths(user_id):
    """Get relative paths to visualizations."""
    return {
        'efficiency_trend': f'sleep_efficiency_trend.png',
        'duration_trend': f'sleep_duration_trend.png',
        'weekday_weekend_duration': f'weekday_weekend_duration.png',
        'weekday_weekend_bedtime': f'weekday_weekend_bedtime.png',
        'correlation_heatmap': f'sleep_metrics_correlation.png',
        'quality_distribution': f'sleep_quality_distribution.png'
    }


def _generate_html_content(user_profile, formatted_metrics, recommendations, visualization_paths):
    """Generate HTML content for the report."""
    # Basic user info
    user_id = user_profile['user_id']
    profession = user_profile['profession']
    age = user_profile['age']
    region = user_profile.get('region', 'Unknown')
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sleep Insights for User {user_id}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #3a86ff;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .section {{
                margin-bottom: 30px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
            }}
            .metrics-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
            }}
            .metric-card {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                min-width: 180px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric-title {{
                font-size: 0.9rem;
                color: #666;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 1.4rem;
                font-weight: bold;
                color: #333;
            }}
            .recommendation {{
                background-color: #f0f7ff;
                border-left: 5px solid #3a86ff;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 0 5px 5px 0;
            }}
            .recommendation-title {{
                font-weight: bold;
                margin-bottom: 5px;
                color: #2b5797;
            }}
            .chart-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: space-between;
            }}
            .chart {{
                margin-bottom: 20px;
                text-align: center;
            }}
            .chart img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Sleep Insights Report</h1>
            <p>User: {user_id} | Age: {age} | Profession: {profession} | Region: {region}</p>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="section">
            <h2>Sleep Summary</h2>
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-title">Average Sleep Efficiency</div>
                    <div class="metric-value">{formatted_metrics['avg_sleep_efficiency']}</div>
                    {f"<div>Trend: {formatted_metrics['efficiency_trend']}</div>" if 'efficiency_trend' in formatted_metrics else ""}
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Average Sleep Duration</div>
                    <div class="metric-value">{formatted_metrics['avg_sleep_duration']}</div>
                    {f"<div>Trend: {formatted_metrics['duration_trend']}</div>" if 'duration_trend' in formatted_metrics else ""}
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Time to Fall Asleep</div>
                    <div class="metric-value">{formatted_metrics['avg_sleep_onset']}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Sleep Quality Rating</div>
                    <div class="metric-value">{formatted_metrics['avg_sleep_rating']}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Sleep Trends</h2>
            <div class="chart-container">
                <div class="chart">
                    <h3>Sleep Efficiency Over Time</h3>
                    <img src="{visualization_paths['efficiency_trend']}" alt="Sleep Efficiency Trend">
                </div>
                <div class="chart">
                    <h3>Sleep Duration Over Time</h3>
                    <img src="{visualization_paths['duration_trend']}" alt="Sleep Duration Trend">
                </div>
            </div>
        </div>
    """
    
    # Add weekday/weekend comparison if available
    if 'weekday_weekend_duration' in visualization_paths:
        html_content += f"""
        <div class="section">
            <h2>Weekday vs Weekend Patterns</h2>
            <div class="chart-container">
                <div class="chart">
                    <h3>Sleep Duration Comparison</h3>
                    <img src="{visualization_paths['weekday_weekend_duration']}" alt="Weekday vs Weekend Sleep Duration">
                </div>
                <div class="chart">
                    <h3>Bedtime Comparison</h3>
                    <img src="{visualization_paths['weekday_weekend_bedtime']}" alt="Weekday vs Weekend Bedtime">
                </div>
            </div>
        </div>
        """
    
    # Add correlation analysis
    html_content += f"""
        <div class="section">
            <h2>Sleep Metrics Correlation</h2>
            <div class="chart">
                <img src="{visualization_paths['correlation_heatmap']}" alt="Sleep Metrics Correlation">
                <p>This heatmap shows how different sleep metrics are related to each other. Strong positive correlations are shown in dark blue, while negative correlations are shown in dark red.</p>
            </div>
        </div>
    """
    
    # Add recommendations section
    html_content += """
        <div class="section">
            <h2>Personalized Recommendations</h2>
    """
    
    # Add each recommendation
    for rec in recommendations:
        html_content += f"""
            <div class="recommendation">
                <div class="recommendation-title">{rec['title']}</div>
                <p>{rec['content']}</p>
            </div>
        """
    
    # Close recommendations section
    html_content += """
        </div>
    """
    
    # Close HTML document
    html_content += """
        <div class="section">
            <h2>Next Steps</h2>
            <p>Continue tracking your sleep patterns to receive more personalized insights. Remember that sleep improvement is a gradual process, and small changes can lead to significant improvements over time.</p>
            <p>Check back regularly for updated analysis and recommendations based on your latest sleep data.</p>
        </div>
    </body>
    </html>
    """
    
    return html_content


def create_markdown_report(user_profile, sleep_metrics, recommendations, visualization_dir, output_dir):
    """Create a markdown version of the sleep insights report."""
    user_id = user_profile['user_id']
    report_path = os.path.join(output_dir, f"{user_id}_sleep_insights.md")
    
    # Format metrics for display
    formatted_metrics = _format_metrics_for_display(sleep_metrics)
    
    # Create markdown content
    md_content = f"""
# Sleep Insights Report for {user_id}

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**User:** {user_id}  
**Age:** {user_profile['age']}  
**Profession:** {user_profile['profession']}  
**Region:** {user_profile.get('region', 'Unknown')}

## Sleep Summary

- **Average Sleep Efficiency:** {formatted_metrics['avg_sleep_efficiency']}
- **Average Sleep Duration:** {formatted_metrics['avg_sleep_duration']}
- **Average Time to Fall Asleep:** {formatted_metrics['avg_sleep_onset']}
- **Average Sleep Quality Rating:** {formatted_metrics['avg_sleep_rating']}

## Sleep Trends

![Sleep Efficiency Trend](sleep_efficiency_trend.png)
![Sleep Duration Trend](sleep_duration_trend.png)

## Personalized Recommendations

"""
    
    # Add each recommendation
    for rec in recommendations:
        md_content += f"""
### {rec['title']}
{rec['content']}

"""
    
    # Add next steps
    md_content += """
## Next Steps

Continue tracking your sleep patterns to receive more personalized insights. Remember that sleep improvement is a gradual process, and small changes can lead to significant improvements over time.

Check back regularly for updated analysis and recommendations based on your latest sleep data.
"""
    
    # Write to file
    with open(report_path, 'w') as f:
        f.write(md_content)
    
    return report_path