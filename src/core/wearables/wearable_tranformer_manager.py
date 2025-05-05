"""
Wearable transformer manager that selects the appropriate transformer
based on device type and handles the data transformation process.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Union

from src.core.wearables.wearable_base_transformer import BaseWearableTransformer
from src.core.wearables.apple_watch_transformer import AppleWatchTransformer
from src.core.wearables.fitbit_transformer import FitbitTransformer
from src.core.wearables.samsung_galaxy_watch_transformer import SamsungWatchTransformer
from src.core.wearables.google_watch_transformer import GoogleWatchTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/wearable_transformer.log')
    ]
)

logger = logging.getLogger(__name__)

class WearableTransformerManager:
    """Manager class for wearable data transformers"""
    
    def __init__(self):
        """Initialize the transformer manager with all available transformers"""
        self.transformers = {
            'apple_watch': AppleWatchTransformer(),
            'fitbit': FitbitTransformer(),
            'samsung_watch': SamsungWatchTransformer(),
            'google_watch': GoogleWatchTransformer()
        }
        
        # Map alternative device names to standard names
        self.device_aliases = {
            'apple': 'apple_watch',
            'iwatch': 'apple_watch',
            'apple watch': 'apple_watch',
            'fitbit': 'fitbit',
            'fit bit': 'fitbit',
            'samsung': 'samsung_watch',
            'galaxy watch': 'samsung_watch',
            'galaxy': 'samsung_watch',
            'samsung galaxy watch': 'samsung_watch',
            'google': 'google_watch',
            'pixel watch': 'google_watch',
            'google pixel': 'google_watch',
            'google pixel watch': 'google_watch'
        }
    
    def transform_data(self, 
                      data: Union[pd.DataFrame, List[Dict], Dict], 
                      device_type: str,
                      users_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transform wearable data to standardized sleep format using the appropriate transformer
        
        Args:
            data: Wearable data as DataFrame, list of dicts, or single dict
            device_type: Type of wearable device
            users_df: Optional DataFrame with user information
            
        Returns:
            DataFrame with standardized sleep data
        """
        # Normalize device type
        normalized_device = self._normalize_device_type(device_type)
        
        if normalized_device not in self.transformers:
            logger.error(f"Unsupported device type: {device_type} (normalized to {normalized_device})")
            raise ValueError(f"Unsupported device type: {device_type}")
        
        logger.info(f"Using transformer for device type: {normalized_device}")
        
        # Get the appropriate transformer
        transformer = self.transformers[normalized_device]
        
        # Transform the data
        try:
            transformed_data = transformer.transform(data, users_df)
            return transformed_data
        except Exception as e:
            logger.error(f"Error transforming {normalized_device} data: {str(e)}")
            raise
    
    def _normalize_device_type(self, device_type: str) -> str:
        """Normalize device type string to standard format"""
        if not device_type:
            raise ValueError("Device type cannot be empty")
            
        # Convert to lowercase and strip whitespace
        normalized = device_type.lower().strip()
        
        # Check if it's in aliases
        if normalized in self.device_aliases:
            return self.device_aliases[normalized]
        
        # Check if it's already a valid device type
        if normalized in self.transformers:
            return normalized
        
        # Try to match partial names
        for alias, standard_name in self.device_aliases.items():
            if alias in normalized or normalized in alias:
                return standard_name
        
        # Default to the input if no match found
        logger.warning(f"Unrecognized device type: {device_type}, using as is")
        return normalized
    
    def get_supported_devices(self) -> List[str]:
        """Get list of supported device types"""
        return list(self.transformers.keys())
    
    def add_custom_transformer(self, device_type: str, transformer: BaseWearableTransformer) -> None:
        """
        Add a custom transformer for a new device type
        
        Args:
            device_type: Name of the device type
            transformer: Instance of a transformer that inherits from BaseWearableTransformer
        """
        if not isinstance(transformer, BaseWearableTransformer):
            raise TypeError("Transformer must be a subclass of BaseWearableTransformer")
            
        normalized_device = device_type.lower().strip()
        self.transformers[normalized_device] = transformer
        logger.info(f"Added custom transformer for device type: {normalized_device}")
    
    def batch_transform(self, 
                       data_frames: Dict[str, pd.DataFrame], 
                       users_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transform multiple device data sources and combine them
        
        Args:
            data_frames: Dictionary mapping device types to DataFrames
            users_df: Optional DataFrame with user information
            
        Returns:
            Combined DataFrame with standardized sleep data
        """
        all_transformed = []
        
        for device_type, data_df in data_frames.items():
            try:
                transformed = self.transform_data(data_df, device_type, users_df)
                all_transformed.append(transformed)
                logger.info(f"Successfully transformed {len(transformed)} records for {device_type}")
            except Exception as e:
                logger.error(f"Error transforming data for {device_type}: {str(e)}")
                continue
        
        if not all_transformed:
            logger.warning("No data was transformed successfully")
            return pd.DataFrame()
        
        # Combine all transformed DataFrames
        combined_df = pd.concat(all_transformed, ignore_index=True)
        
        # Remove duplicate sleep records (same user, same date)
        if 'user_id' in combined_df.columns and 'date' in combined_df.columns:
            # Sort by device quality (higher quality devices first)
            device_quality = {
                'apple_watch': 1,  # Highest quality
                'samsung_watch': 2,
                'google_watch': 3,
                'fitbit': 4
            }
            
            # Add a quality column for sorting
            combined_df['device_quality'] = combined_df['device_type'].map(
                lambda x: device_quality.get(x, 99)  # Unknown devices get lowest quality
            )
            
            # Sort by quality to keep best device data when dropping duplicates
            combined_df = combined_df.sort_values('device_quality')
            
            # Drop duplicates, keeping the first (highest quality) record
            combined_df = combined_df.drop_duplicates(subset=['user_id', 'date'], keep='first')
            
            # Remove the temporary quality column
            combined_df = combined_df.drop(columns=['device_quality'])
        
        return combined_df