import numpy as np
import pandas as pd
import yaml
import os
import uuid
from datetime import datetime, timedelta
from faker import Faker

class UserGenerator:
    def __init__(self, config_path='config/data_generation_config.yaml'):
        print(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.user_config = self.config['user_profiles']
        self.sleep_patterns = self.config['sleep_patterns']
        
        # Initialize Faker for generating realistic user data
        self.faker = Faker()
        
    def generate_users(self, count=None):
        """Generate a specified number of user profiles"""
        if count is None:
            count = self.user_config['count']
        
        users = []
        
        for _ in range(count):
            user = self._generate_single_user()
            users.append(user)
        
        return pd.DataFrame(users)
    
    def _generate_single_user(self):
        """Generate a single user profile"""
        age_range = self.user_config['age_range']
        genders = self.user_config['genders']
        
        # Generate pattern based on distribution
        sleep_pattern_dist = self.user_config['sleep_patterns']
        patterns = list(sleep_pattern_dist.keys())
        weights = list(sleep_pattern_dist.values())
        sleep_pattern = np.random.choice(patterns, p=weights)
        
        # Generate device based on distribution
        device_dist = self.user_config['device_distribution']
        devices = list(device_dist.keys())
        device_weights = list(device_dist.values())
        device = np.random.choice(devices, p=device_weights)
        
        # Generate age with a bias toward working-age adults
        if sleep_pattern == 'shift_worker':
            # Shift workers tend to be working age
            age = np.random.randint(25, 55)
        else:
            age = np.random.randint(age_range[0], age_range[1] + 1)
        
        # Generate profession based on age and sleep pattern
        profession = self._generate_profession(age, sleep_pattern)
        
        # Generate region (city, state/province, country)
        region = self._generate_region()
        
        # Generate data consistency level (how often they might skip logging)
        data_consistency = np.random.beta(12, 2)  # Skewed toward higher consistency
        
        # Generate sleep consistency (regularity of sleep patterns)
        if sleep_pattern == 'variable':
            sleep_consistency = np.random.beta(2, 5)  # Lower consistency
        else:
            sleep_consistency = np.random.beta(5, 2)  # Higher consistency
        
        return {
            'user_id': str(uuid.uuid4())[:8],
            'age': age,
            'gender': np.random.choice(genders),
            'profession': profession,
            'region': region,
            'sleep_pattern': sleep_pattern,
            'device_type': device,
            'data_consistency': data_consistency,
            'sleep_consistency': sleep_consistency,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _generate_profession(self, age, sleep_pattern):
        """Generate profession that makes sense for the user's age and sleep pattern"""
        # For retired users
        if age >= 65 and np.random.random() < 0.7:
            return self.faker.random_element([
                'Retired', 'Retired Teacher', 'Retired Engineer', 
                'Retired Healthcare Worker', 'Retired Business Owner',
                'Part-time Consultant', 'Volunteer'
            ])
        
        # For shift workers, we want professions that often involve shift work
        if sleep_pattern == 'shift_worker':
            return self.faker.random_element([
                'Nurse', 'Doctor', 'Police Officer', 'Firefighter',
                'Paramedic', 'Security Guard', 'Factory Worker',
                'Truck Driver', 'Pilot', 'Flight Attendant',
                'Hotel Staff', 'Call Center Agent', 'Bartender',
                'Restaurant Server', 'Baker', 'Air Traffic Controller',
                'Transit Operator', 'Casino Worker', 'Warehouse Worker'
            ])
        
        # For very young users (18-22), likely students
        if age < 23:
            return self.faker.random_element([
                'Student', 'Part-time Retail Worker', 'Intern',
                'Barista', 'Server', 'Delivery Driver'
            ])
        
        # For insomnia pattern, high-stress jobs are more common
        if sleep_pattern == 'insomnia':
            return self.faker.random_element([
                'Software Engineer', 'Doctor', 'Lawyer', 'Financial Analyst',
                'Executive', 'Manager', 'Consultant', 'Entrepreneur',
                'Journalist', 'Professor', 'Researcher', 'Air Traffic Controller'
            ])
        
        # For oversleeper pattern, less intense jobs
        if sleep_pattern == 'oversleeper':
            return self.faker.random_element([
                'Writer', 'Artist', 'Designer', 'Librarian', 
                'Administrative Assistant', 'Technician', 'Analyst',
                'Remote Worker', 'Freelancer', 'Content Creator'
            ])
            
        # For variable pattern, mix of professions
        if sleep_pattern == 'variable':
            return self.faker.random_element([
                'Sales Representative', 'Marketing Specialist', 'Travel Agent',
                'Photographer', 'Journalist', 'Consultant', 'Event Planner',
                'Freelancer', 'Entrepreneur', 'Social Media Manager'
            ])
        
        # For everyone else, generate a realistic profession
        return self.faker.job()

    def _generate_region(self):
        """Generate a global region (city, state/province, country)"""
        # Define region distributions
        region_dist = self.user_config.get('region_distribution', {
            'north_america': 0.4,
            'europe': 0.3,
            'asia': 0.2,
            'other': 0.1
        })
        
        regions = list(region_dist.keys())
        weights = list(region_dist.values())
        region_type = np.random.choice(regions, p=weights)
        
        if region_type == 'north_america':
            country = self.faker.random_element(['United States', 'Canada', 'Mexico'])
            if country == 'United States':
                city = self.faker.city()
                state = self.faker.state()
                return f"{city}, {state}, {country}"
            else:
                city = self.faker.city()
                province = self.faker.state()  # Using state as a proxy for province
                return f"{city}, {province}, {country}"
        
        elif region_type == 'europe':
            country = self.faker.random_element([
                'United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 
                'Netherlands', 'Sweden', 'Norway', 'Finland', 'Denmark'
            ])
            city = self.faker.city()
            region = self.faker.state()  # Using state as a proxy for region
            return f"{city}, {region}, {country}"
        
        elif region_type == 'asia':
            country = self.faker.random_element([
                'Japan', 'China', 'South Korea', 'India', 'Singapore', 
                'Thailand', 'Malaysia', 'Vietnam', 'Indonesia'
            ])
            city = self.faker.city()
            region = self.faker.state()  # Using state as a proxy for region
            return f"{city}, {region}, {country}"
            
        else:  # other regions
            city = self.faker.city()
            state = self.faker.state()
            country = self.faker.country()
            return f"{city}, {state}, {country}"