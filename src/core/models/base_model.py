class BaseModel:
    """Base class for all models with common functionality"""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def save(self, path):
        """Save model to path"""
        pass
        
    def load(self, path):
        """Load model from path"""
        pass