# src/core/data/repository.py
class DataRepository:
    """Data access layer for sleep data"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.cache = {}
    
    def get_user_data(self, user_id=None):
        """Get user data, optionally filtered by user_id"""
        # Implement caching
        cache_key = f"user_{user_id}" if user_id else "all_users"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load data
        # ...
        
        # Cache and return
        self.cache[cache_key] = data
        return data