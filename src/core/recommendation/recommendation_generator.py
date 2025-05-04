"""
Module for generating personalized sleep recommendations based on user profiles and sleep data.
"""

from src.utils.constants import profession_categories


def generate_personalized_recommendations(user_profile, sleep_metrics, recommendations=None):
    """
    Generate personalized recommendations based on user profile and sleep data.
    
    Args:
        user_profile: Dictionary containing user profile information
        sleep_metrics: Dictionary of calculated sleep metrics
        recommendations: DataFrame of previous recommendations (optional)
        
    Returns:
        list: List of recommendation dictionaries with category, title, and content
    """
    personalized_recommendations = []
    
    # Extract key information
    profession = user_profile['profession']
    age = user_profile['age']
    sleep_pattern = user_profile['sleep_pattern']
    
    # Extract region information (assuming format "City, State, Country")
    region = "Unknown"
    country = "Unknown"
    
    if 'region' in user_profile and isinstance(user_profile['region'], str):
        region = user_profile['region']
        if ',' in region:
            parts = region.split(',')
            if len(parts) >= 3:
                country = parts[-1].strip()
    
    # Determine profession category
    profession_category = _extract_profession_category(profession)
    
    # Determine region category
    region_category = _extract_region_category(region, country)
    
    # Generate general sleep recommendations
    general_rec = {
        'category': 'general',
        'title': 'General Sleep Recommendation',
        'content': f"Based on your sleep pattern ({sleep_pattern}), aim for consistent sleep and wake times, even on weekends. This helps regulate your body's internal clock."
    }
    personalized_recommendations.append(general_rec)
    
    # Generate profession-specific recommendations
    prof_rec = _generate_profession_recommendation(profession, profession_category)
    personalized_recommendations.append(prof_rec)
    
    # Generate region-specific recommendations
    region_rec = _generate_region_recommendation(region, region_category, country)
    personalized_recommendations.append(region_rec)
    
    # Add data-driven recommendations based on sleep metrics
    data_recs = _generate_data_driven_recommendations(sleep_metrics)
    personalized_recommendations.extend(data_recs)
    
    return personalized_recommendations


def _extract_profession_category(profession):
    """Extract profession category from profession string."""
    for category, keywords in profession_categories.items():
        if any(keyword.lower() in profession.lower() for keyword in keywords):
            return category
    return "other"


def _extract_region_category(region, country):
    """Extract region category from region and country."""
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


def _generate_profession_recommendation(profession, profession_category):
    """Generate profession-specific sleep recommendation."""
    if profession_category == 'healthcare':
        return {
            'category': 'profession',
            'title': 'Healthcare Professional Sleep Tips',
            'content': f"As a {profession}, your shifting work schedule can impact sleep quality. Try using blackout curtains and white noise machines to create a consistent sleep environment regardless of when you sleep."
        }
    elif profession_category == 'tech':
        return {
            'category': 'profession',
            'title': 'Tech Professional Sleep Tips',
            'content': f"Your profession likely involves significant screen time. Consider using blue light filters on all devices, especially in the evening, and try to disconnect from screens at least 1 hour before bedtime."
        }
    elif profession_category == 'service':
        return {
            'category': 'profession',
            'title': 'Service Industry Sleep Tips',
            'content': f"Service positions like yours often involve variable schedules and potentially stressful interactions. Try a 10-minute decompression ritual after work to mentally separate work stress from sleep time."
        }
    elif profession_category == 'education':
        return {
            'category': 'profession',
            'title': 'Educator Sleep Tips',
            'content': f"As an educator, work stress and take-home work can affect sleep. Set clear boundaries between work and personal time. Try to finish grading or preparation at least 2 hours before bedtime."
        }
    else:
        return {
            'category': 'profession',
            'title': f'Sleep Tips for {profession}',
            'content': f"Consider how your work as a {profession} impacts your sleep schedule and stress levels. Create a transition routine between work and sleep to help your mind unwind."
        }


def _generate_region_recommendation(region, region_category, country):
    """Generate region-specific sleep recommendation."""
    if region_category == 'north_america':
        return {
            'category': 'region',
            'title': 'Regional Sleep Consideration',
            'content': f"In {country}, many people struggle with work-life balance. Consider setting clear boundaries on work hours and notifications to protect your sleep time."
        }
    elif region_category == 'europe':
        return {
            'category': 'region',
            'title': 'Regional Sleep Consideration',
            'content': f"In many European countries like {country}, later dinner times can impact sleep quality. Try to eat your last meal at least 3 hours before bedtime for better sleep quality."
        }
    elif region_category == 'asia':
        return {
            'category': 'region',
            'title': 'Regional Sleep Consideration',
            'content': f"In {country}, urban light pollution and population density can affect sleep quality. Consider using room-darkening curtains and white noise to create an optimal sleep environment."
        }
    else:
        return {
            'category': 'region',
            'title': 'Regional Sleep Consideration',
            'content': f"Consider how the cultural norms and environment in {region} might be affecting your sleep, including meal times, social expectations, and climate factors."
        }


def _generate_data_driven_recommendations(sleep_metrics):
    """Generate data-driven recommendations based on sleep metrics."""
    recommendations = []
    
    # Recommendation for sleep onset time
    if sleep_metrics['avg_sleep_onset'] > 30:
        recommendations.append({
            'category': 'data',
            'title': 'Improve Sleep Onset Time',
            'content': f"You're taking an average of {sleep_metrics['avg_sleep_onset']:.1f} minutes to fall asleep. Try a relaxation technique like deep breathing or progressive muscle relaxation to reduce sleep onset time."
        })
    
    # Recommendation for bedtime consistency
    if 'bedtime_difference' in sleep_metrics and sleep_metrics['bedtime_difference'] > 1.5:
        recommendations.append({
            'category': 'data',
            'title': 'Improve Bedtime Consistency',
            'content': f"Your weekend bedtime differs from weekday by {sleep_metrics['bedtime_difference']:.1f} hours. Try to keep this difference under 1 hour to prevent 'social jet lag' and improve overall sleep quality."
        })
    
    # Recommendation for night awakenings
    if sleep_metrics['avg_awakenings'] > 2:
        recommendations.append({
            'category': 'data',
            'title': 'Reduce Night Awakenings',
            'content': f"You're experiencing about {sleep_metrics['avg_awakenings']:.1f} awakenings per night. Consider factors like room temperature, noise, or light that might be disrupting your sleep. A cooler, darker, and quieter environment often helps reduce awakenings."
        })
    
    # Recommendation based on recent trend
    if 'efficiency_change' in sleep_metrics:
        if sleep_metrics['efficiency_change'] > 0.05:
            recommendations.append({
                'category': 'trend',
                'title': 'Recent Improvement',
                'content': f"Your sleep efficiency has improved by {sleep_metrics['efficiency_change']*100:.1f}% in the last week compared to the previous week. Whatever changes you've made recently appear to be working well!"
            })
        elif sleep_metrics['efficiency_change'] < -0.05:
            recommendations.append({
                'category': 'trend',
                'title': 'Recent Decline',
                'content': f"Your sleep efficiency has declined by {abs(sleep_metrics['efficiency_change']*100):.1f}% in the last week. Consider what factors might have changed recently and try to address them."
            })
        else:
            recommendations.append({
                'category': 'trend',
                'title': 'Stable Patterns',
                'content': f"Your sleep patterns have been relatively stable recently. To further improve, try introducing one new sleep hygiene practice this week."
            })
    
    return recommendations