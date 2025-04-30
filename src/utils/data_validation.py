# src/utils/data_validation.py

from pydantic import BaseModel, Field, ValidationError, validator
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
import logging

logger = logging.getLogger(__name__)


class FileValidator:
    """Utility class for validating external data files"""
    
    @staticmethod
    def validate_csv(file_path, model_class, error_handling='warn'):
        """
        Validate a CSV file against a Pydantic model
        
        Args:
            file_path: Path to CSV file
            model_class: Pydantic model class to validate against
            error_handling: 'warn', 'raise', or 'filter'
        
        Returns:
            DataFrame with validated data (if error_handling='filter') 
            or original data (if error_handling='warn')
        """
        # Check if file exists
        if not os.path.exists(file_path):
            if error_handling == 'raise':
                raise FileNotFoundError(f"File not found: {file_path}")
            logger.error(f"File not found: {file_path}")
            return None
        
        # Read CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            if error_handling == 'raise':
                raise e
            logger.error(f"Error reading CSV file: {e}")
            return None
        
        if error_handling == 'filter':
            # Filter out invalid rows
            valid_rows = []
            for i, row in df.iterrows():
                try:
                    model_class(**row.to_dict())
                    valid_rows.append(i)
                except ValidationError as e:
                    logger.warning(f"Validation error in row {i}: {e}")
            
            # Return only valid rows
            return df.loc[valid_rows]
        
        elif error_handling == 'warn':
            # Validate but don't filter
            for i, row in df.iterrows():
                try:
                    model_class(**row.to_dict())
                except ValidationError as e:
                    logger.warning(f"Validation error in row {i}: {e}")
            
            # Return original data
            return df
        
        else:  # 'raise'
            # Validate all rows, raise on first error
            for i, row in df.iterrows():
                model_class(**row.to_dict())
            
            # Return validated data
            return df
    
    @staticmethod
    def validate_json(file_path, model_class, error_handling='warn'):
        """
        Validate a JSON file against a Pydantic model
        
        Args:
            file_path: Path to JSON file
            model_class: Pydantic model class to validate against
            error_handling: 'warn', 'raise', or 'filter'
        
        Returns:
            Dictionary with validated data or List of validated items
        """
        # Check if file exists
        if not os.path.exists(file_path):
            if error_handling == 'raise':
                raise FileNotFoundError(f"File not found: {file_path}")
            logger.error(f"File not found: {file_path}")
            return None
        
        # Read JSON file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            if error_handling == 'raise':
                raise e
            logger.error(f"Error reading JSON file: {e}")
            return None
        
        # Handle different JSON structures
        if isinstance(data, list):
            if error_handling == 'filter':
                # Filter out invalid items
                valid_items = []
                for i, item in enumerate(data):
                    try:
                        validated = model_class(**item)
                        valid_items.append(validated.dict())
                    except ValidationError as e:
                        logger.warning(f"Validation error in item {i}: {e}")
                return valid_items
            
            elif error_handling == 'warn':
                # Validate but don't filter
                for i, item in enumerate(data):
                    try:
                        model_class(**item)
                    except ValidationError as e:
                        logger.warning(f"Validation error in item {i}: {e}")
                return data
            
            else:  # 'raise'
                # Validate all items, raise on first error
                return [model_class(**item).dict() for item in data]
        
        else:  # data is a dictionary
            if error_handling == 'raise' or error_handling == 'filter':
                # Validate and return
                try:
                    validated = model_class(**data)
                    return validated.dict()
                except ValidationError as e:
                    if error_handling == 'raise':
                        raise e
                    logger.error(f"Validation error in JSON: {e}")
                    return None
            
            else:  # 'warn'
                # Validate but don't filter
                try:
                    model_class(**data)
                except ValidationError as e:
                    logger.warning(f"Validation error in JSON: {e}")
                return data