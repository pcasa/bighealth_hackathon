"""
Constants used throughout the Sleep Insights App.
This includes categorization dictionaries, default values, and other constants.
"""

# Profession categories with associated keywords for categorization
profession_categories = {
    'healthcare': [
        'doctor', 'nurse', 'physician', 'therapist', 'pharmacist', 'medical', 
        'healthcare', 'hospital', 'clinical', 'dentist', 'veterinarian'
    ],
    'tech': [
        'software', 'developer', 'engineer', 'programmer', 'analyst', 'IT', 
        'tech', 'computer', 'data', 'system', 'network', 'web', 'product'
    ],
    'service': [
        'retail', 'server', 'customer', 'service', 'hospitality', 'food', 
        'restaurant', 'hotel', 'store', 'barista', 'cashier', 'sales', 'chef'
    ],
    'education': [
        'teacher', 'professor', 'educator', 'tutor', 'school', 'university', 
        'college', 'academic', 'instructor', 'teaching', 'education', 'faculty'
    ],
    'industrial': [
        'factory', 'manufacturing', 'construction', 'worker', 'labor', 'machine', 
        'operator', 'production', 'welder', 'mechanic', 'technician', 'maintenance'
    ],
    'office': [
        'manager', 'administrator', 'executive', 'clerical', 'assistant', 'office', 
        'secretary', 'coordinator', 'accountant', 'finance', 'hr', 'administrative'
    ]
}

# Region categories for demographic analysis
region_categories = {
    'north_america': [
        'united states', 'usa', 'us', 'canada', 'mexico', 'america', 'north america'
    ],
    'europe': [
        'uk', 'united kingdom', 'england', 'france', 'germany', 'italy', 'spain', 
        'europe', 'european', 'netherlands', 'sweden', 'norway', 'finland'
    ],
    'asia': [
        'china', 'japan', 'india', 'korea', 'singapore', 'thailand', 'asia', 
        'asian', 'philippines', 'indonesia', 'malaysia', 'vietnam'
    ],
    'other': [
        'australia', 'brazil', 'argentina', 'south africa', 'africa', 'middle east',
        'new zealand', 'russia', 'uae', 'saudi', 'egypt', 'nigeria'
    ]
}

# Sleep pattern descriptions for UI and reporting
sleep_pattern_descriptions = {
    'normal': 'Regular sleep patterns with consistent sleep-wake times and good sleep efficiency',
    'insomnia': 'Difficulty falling or staying asleep, with frequent awakenings and reduced sleep quality',
    'shift_worker': 'Irregular sleep schedule due to rotating shifts, with daytime sleep periods',
    'oversleeper': 'Extended time in bed and sleep duration, often with excessive daytime sleepiness',
    'variable': 'Highly inconsistent sleep patterns with significant night-to-night variability'
}

# Sleep stage descriptions
sleep_stage_descriptions = {
    'deep': 'Deep sleep (N3) - The most restorative stage where tissue growth and repair occurs',
    'light': 'Light sleep (N1/N2) - Transitional sleep stages where body temperature drops and heart rate slows',
    'rem': 'REM sleep - Associated with dreaming, memory consolidation, and emotional processing',
    'awake': 'Awake - Brief periods of wakefulness during the night'
}

# Default values for data filtering and analysis
default_values = {
    'min_sleep_duration': 3.0,  # Minimum valid sleep duration in hours
    'max_sleep_duration': 14.0,  # Maximum valid sleep duration in hours
    'min_sleep_efficiency': 0.5,  # Minimum valid sleep efficiency
    'max_subjective_rating': 10,  # Maximum subjective rating value
    'trend_analysis_window': 14,  # Default number of days for trend analysis
    'consistency_threshold': 0.7,  # Threshold for determining consistent sleep schedule
    'insomnia_awakenings_threshold': 3,  # Minimum awakenings to qualify for insomnia pattern
    'severe_insomnia_threshold': 3,  # Number of no-sleep nights to qualify as severe insomnia
}

# Message category mappings for recommendation engine
message_categories = {
    'onboarding': 'For new users with less than 7 days of data',
    'strong_encouragement': 'For users with significant (>10%) improvement in sleep metrics',
    'encouragement': 'For users with moderate (5-10%) improvement in sleep metrics',
    'support': 'For users with declining sleep metrics who had better sleep before',
    'gentle_reminder': 'For users with slight decline in sleep metrics',
    'maintenance': 'For users with stable sleep metrics within ±2% variation',
    'consistency_reminder': 'For users with inconsistent tracking behavior',
    'moderate_insomnia': 'For users reporting multiple nights of poor sleep',
    'severe_insomnia': 'For users reporting 3+ nights without sleep in a 14-day period'
}