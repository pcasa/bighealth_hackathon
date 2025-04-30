# src/config/config_manager.py
class ConfigManager:
    """Central configuration manager"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path or 'config/config.yaml'
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get(self, key, default=None):
        """Get configuration value"""
        # Support nested keys with dot notation
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value