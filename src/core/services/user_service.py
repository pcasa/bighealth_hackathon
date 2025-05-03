# src/core/services/user_service.py
import uuid
from datetime import datetime

class UserService:
    """Service layer for user-related operations"""
    
    def __init__(self, repository):
        self.repository = repository
    
    async def get_all_users(self, limit=100):
        """Get all users with limit"""
        users_df = self.repository.get_user_data()
        
        if len(users_df) > limit:
            users_df = users_df.head(limit)
            
        return users_df.to_dict('records')
    
    async def get_user(self, user_id):
        """Get a user by ID"""
        user_df = self.repository.get_user_data(user_id)
        
        if len(user_df) == 0:
            return None
            
        return user_df.iloc[0].to_dict()
    
    async def create_user(self, user_data):
        """Create a new user"""
        # Generate user ID
        user_id = str(uuid.uuid4())[:8]
        
        # Add default fields
        user_data['user_id'] = user_id
        user_data['data_consistency'] = 0.85
        user_data['sleep_consistency'] = 0.75
        user_data['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save user
        return self.repository.save_user(user_data)
    
    async def update_user(self, user_id, update_data):
        """Update a user"""
        # Get existing user
        user_df = self.repository.get_user_data(user_id)
        
        if len(user_df) == 0:
            return None
            
        # Update fields
        user_data = user_df.iloc[0].to_dict()
        
        for key, value in update_data.items():
            if value is not None:  # Only update non-None values
                user_data[key] = value
        
        # Save updated user
        return self.repository.save_user(user_data)
    
    async def delete_user(self, user_id):
        """Delete a user"""
        return self.repository.delete_user(user_id)
    
    async def get_profession_stats(self):
        """Get sleep statistics by profession"""
        users_df = self.repository.get_user_data()
        sleep_df = self.repository.get_sleep_data()
        
        if len(users_df) == 0 or len(sleep_df) == 0:
            return []
        
        # Ensure profession_category exists
        if 'profession_category' not in users_df.columns:
            # Categorize professions
            profession_categories = {
                'healthcare': ['doctor', 'nurse', 'physician', 'therapist', 'medical'],
                'tech': ['developer', 'engineer', 'programmer', 'analyst', 'IT'],
                'service': ['retail', 'server', 'customer', 'service', 'hospitality'],
                'education': ['teacher', 'professor', 'educator', 'tutor', 'school'],
                'office': ['manager', 'administrator', 'executive', 'clerical', 'assistant']
            }
            
            users_df['profession_category'] = 'other'
            for category, keywords in profession_categories.items():
                mask = users_df['profession'].apply(lambda x: any(kw.lower() in str(x).lower() for kw in keywords))
                users_df.loc[mask, 'profession_category'] = category
        
        # Merge user and sleep data
        merged_df = pd.merge(sleep_df, users_df[['user_id', 'profession_category']], on='user_id')
        
        # Calculate statistics by profession
        prof_stats = merged_df.groupby('profession_category').agg({
            'sleep_efficiency': 'mean',
            'sleep_duration_hours': 'mean',
            'subjective_rating': 'mean',
            'user_id': 'count'
        }).reset_index()
        
        # Format for API response
        result = []
        for _, row in prof_stats.iterrows():
            result.append({
                "profession": row['profession_category'],
                "count": int(row['user_id']),
                "avg_sleep_efficiency": float(row['sleep_efficiency']),
                "avg_sleep_duration": float(row['sleep_duration_hours']),
                "avg_subjective_rating": float(row['subjective_rating'])
            })
        
        return result