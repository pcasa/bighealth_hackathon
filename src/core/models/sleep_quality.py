import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import os
import yaml
import json
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union

from src.core.models.generate_model_card import generate_model_card_with_samples
from src.core.models.improved_sleep_score import ImprovedSleepScoreCalculator



class SleepQualityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(SleepQualityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sleep_score_calculator = ImprovedSleepScoreCalculator()
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Apply fully connected layer
        out = self.fc(out)
        
        return out

class ModelInput(BaseModel):
    """Validates input data for the sleep quality model"""
    # Required fields
    sleep_efficiency: float = Field(..., ge=0.0, le=1.0)
    sleep_duration_hours: float = Field(..., ge=0.0, le=24.0)
    
    # Optional fields with validation
    sleep_onset_latency_minutes: Optional[float] = Field(None, ge=0.0)
    deep_sleep_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    rem_sleep_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    awakenings_count: Optional[int] = Field(None, ge=0)
    
    # Add more fields as needed


class ModelOutput(BaseModel):
    """Validates and structures the model output"""
    predicted_sleep_efficiency: float = Field(..., ge=0.0, le=1.0)
    prediction_confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Optional additional outputs
    sleep_score: Optional[int] = Field(None, ge=0, le=100)
    component_scores: Optional[Dict[str, float]] = Field(None)

class SleepQualityModel:
    def __init__(self, config_path='src/config/model_config.yaml'):
        """Initialize the sleep quality model"""
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.config = config['sleep_quality_model']
        
        self.model = None
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    """
    Update to the preprocess_data method in SleepQualityModel class to handle potential issues
    with user_id and date fields. Insert this updated method into your sleep_quality.py file.
    """

    def preprocess_data(self, data, sequence_length=7):
        """Preprocess data for the model with improved error handling and support for demographic factors"""
        print("Starting preprocessing with data shape:", data.shape)
        
        # Verify required columns are present
        required_columns = ['user_id', 'date']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            print(error_msg)
            raise KeyError(error_msg)
        
        # Ensure date is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            print("Converting date column to datetime")
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            
            # Check if conversion created NaT values
            if data['date'].isna().any():
                print("Warning: Some dates couldn't be converted to datetime")
                # Fill NaT with dummy dates to avoid crashes
                date_nas = data['date'].isna()
                if date_nas.any():
                    base_date = datetime(2024, 1, 1)
                    dummy_dates = [base_date + timedelta(days=i) for i in range(sum(date_nas))]
                    data.loc[date_nas, 'date'] = dummy_dates
        
        # Add season feature based on month
        data['month'] = data['date'].dt.month
        data['season'] = self._get_season(data['month'])
        
        # Add seasonal one-hot encoding
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        for season in seasons:
            data[f'season_{season}'] = (data['season'] == season).astype(float)
        
        # Process age if available
        if 'age' in data.columns:
            # Add age-range features
            data['age_range'] = self._get_age_range(data['age'])
            
            # Add normalized age
            data['age_normalized'] = data['age'] / 100.0  # Simple normalization
        
        # Process profession if available
        if 'profession' in data.columns and 'profession_category' not in data.columns:
            data['profession_category'] = data['profession'].apply(self._categorize_profession)
    
                    
        # Ensure features are in the right format
        required_features = self.config['features']
        available_features = [col for col in required_features if col in data.columns]

        # Ensure profession_category is properly encoded
        if 'profession_category' in data.columns:
            # If it's not already one-hot encoded
            if pd.api.types.is_object_dtype(data['profession_category']):
                print("Converting profession_category to one-hot encoding")
                # Create profession one-hot encoding
                prof_categories = ['healthcare', 'tech', 'service', 'education', 'office', 'other']
                for category in prof_categories:
                    col_name = f'profession_{category}'
                    if col_name not in data.columns:
                        # Use string comparison to avoid issues with numeric values
                        data[col_name] = (data['profession_category'].astype(str) == category).astype(float)
                
                # Remove the original category column from features
                if 'profession_category' in available_features:
                    available_features.remove('profession_category')
        
        # Add demographic and seasonal features if available
        demographic_features = [col for col in data.columns if (
            col.startswith('season_') or 
            col.startswith('profession_') or 
            col == 'age_normalized'
        )]
        
        available_features.extend([f for f in demographic_features if f not in available_features])

        print(f"Available features before filtering: {available_features}")

        for feature in list(available_features):  # Use list() to avoid modifying during iteration
            if not pd.api.types.is_numeric_dtype(data[feature]):
                print(f"Converting non-numeric feature: {feature}")
                
                # For profession_category
                if feature == 'profession_category':
                    # Create one-hot encoding
                    prof_categories = ['healthcare', 'tech', 'service', 'education', 'office', 'other']
                    for category in prof_categories:
                        col_name = f'profession_{category}'
                        if col_name not in data.columns:
                            data[col_name] = (data['profession_category'].astype(str) == category).astype(float)
                        if col_name not in available_features:
                            available_features.append(col_name)
                    
                    # Remove original category column
                    available_features.remove(feature)
                else:
                    # Remove other non-numeric features
                    available_features.remove(feature)

        print(f"Available features after processing: {available_features}")
        # Group by user to create sequences
        try:
            user_groups = data.groupby('user_id')
            print(f"Found {len(user_groups)} user groups")
        except Exception as e:
            print(f"Error grouping by user_id: {str(e)}")
            print(f"user_id column sample: {data['user_id'].head()}")
            print(f"user_id column type: {data['user_id'].dtype}")
            raise
        
        sequences = []
        targets = []
        user_ids = []
        dates = []
        
        for user_id, group in user_groups:
            # Sort by date
            group = group.sort_values('date')
            
            if len(group) <= sequence_length:
                print(f"Warning: User {user_id} has only {len(group)} records, need at least {sequence_length+1}")
                continue
            
            try:
                # Convert to numpy for easier slicing
                user_data = group[available_features].values
                
                # Create sequences
                for i in range(len(user_data) - sequence_length):
                    seq = user_data[i:i+sequence_length]
                    
                    # Ensure sleep_efficiency exists
                    if 'sleep_efficiency' not in group.columns:
                        print(f"Error: 'sleep_efficiency' column not found for user {user_id}")
                        continue
                        
                    target = group.iloc[i+sequence_length]['sleep_efficiency']
                    
                    sequences.append(seq)
                    targets.append(target)
                    user_ids.append(user_id)
                    dates.append(group.iloc[i+sequence_length]['date'])
            except Exception as e:
                print(f"Error processing user {user_id}: {str(e)}")
                continue
        
        if not sequences:
            error_msg = "No valid sequences could be created. Check your data."
            print(error_msg)
            raise ValueError(error_msg)
        
        print(f"Created {len(sequences)} sequences")

        # Debug: Check for non-numeric values in available_features
        for feature in available_features:
            if not pd.api.types.is_numeric_dtype(data[feature]):
                print(f"WARNING: Non-numeric data in feature '{feature}': {data[feature].head()}")
                # Try to convert to numeric if possible
                data[feature] = pd.to_numeric(data[feature], errors='coerce')
                # Fill NaN values after conversion
                data[feature] = data[feature].fillna(0)
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(sequences))
        y = torch.FloatTensor(np.array(targets)).view(-1, 1)
        
        return X, y, user_ids, dates, available_features

    # Helper methods for demographic and seasonal features
    def _get_season(self, month):
        """Determine season from month (Northern Hemisphere)"""
        if isinstance(month, pd.Series):
            return month.apply(self._get_season)
        
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    def _get_age_range(self, age):
        """Convert age to categorical range"""
        if isinstance(age, pd.Series):
            return age.apply(self._get_age_range)
        
        if age < 30:
            return 'young_adult'
        elif age < 50:
            return 'middle_age'
        elif age < 65:
            return 'senior'
        else:
            return 'elderly'

    def _categorize_profession(self, profession):
        """Categorize profession based on keywords"""
        from src.utils.constants import profession_categories
        
        for category, keywords in profession_categories.items():
            if any(keyword.lower() in profession.lower() for keyword in keywords):
                return category
        return "other"

    # Add this method to your SleepQualityModel class
    def train_with_limited_data(self, data):
        """Alternative training approach when there isn't enough sequence data"""
        print("Using alternative training method for limited data")
        
        # Ensure required features are present
        required_features = self.config['features']
        available_features = [col for col in required_features if col in data.columns]
        
        if len(available_features) < len(required_features):
            missing = set(required_features) - set(available_features)
            print(f"Warning: Missing features: {missing}")
            
            # Create dummy columns for missing features
            for missing_feature in missing:
                print(f"Creating dummy column for missing feature: {missing_feature}")
                if 'normalized' in missing_feature:
                    # For normalized features, use appropriate scaling
                    if missing_feature == 'age_normalized' and 'age' in data.columns:
                        data[missing_feature] = data['age'] / 100.0
                    else:
                        data[missing_feature] = 0.5  # Default to mid-range
                elif 'profession_' in missing_feature:
                    # For profession one-hot features
                    data[missing_feature] = 0.0
                elif 'season_' in missing_feature:
                    # For season one-hot features
                    if missing_feature == 'season_Winter' and 'month' in data.columns:
                        data[missing_feature] = data['month'].apply(lambda m: 1.0 if m in [12, 1, 2] else 0.0)
                    elif missing_feature == 'season_Spring' and 'month' in data.columns:
                        data[missing_feature] = data['month'].apply(lambda m: 1.0 if m in [3, 4, 5] else 0.0)
                    elif missing_feature == 'season_Summer' and 'month' in data.columns:
                        data[missing_feature] = data['month'].apply(lambda m: 1.0 if m in [6, 7, 8] else 0.0)
                    elif missing_feature == 'season_Fall' and 'month' in data.columns:
                        data[missing_feature] = data['month'].apply(lambda m: 1.0 if m in [9, 10, 11] else 0.0)
                    else:
                        data[missing_feature] = 0.0
                else:
                    data[missing_feature] = 0.0  # Default value
                    
                available_features.append(missing_feature)
        
        # Store feature columns
        self.feature_columns = available_features
        
        # Fill NaN values - CRITICAL FIX
        for col in available_features:
            if data[col].isna().any():
                print(f"Filling NaN values in column {col}")
                if data[col].dtype == float or data[col].dtype == int:
                    data[col] = data[col].fillna(0.0)
                else:
                    # For non-numeric columns, fill with most common value
                    data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else "unknown")
        
        # Extract features and target
        X = data[available_features].values
        
        # Ensure sleep_efficiency exists and has no NaN values
        if 'sleep_efficiency' not in data.columns:
            raise ValueError("Target column 'sleep_efficiency' missing from data")
        
        if data['sleep_efficiency'].isna().any():
            print("WARNING: NaN values found in target column 'sleep_efficiency'. Filling with mean.")
            data['sleep_efficiency'] = data['sleep_efficiency'].fillna(data['sleep_efficiency'].mean())
        
        y = data['sleep_efficiency'].values
        
        # Create a simple fully connected network instead of LSTM
        input_size = len(available_features)
        hidden_size = self.config['hyperparameters']['hidden_size']
        
        # Define a simple model
        class SimpleNN(nn.Module):
            def __init__(self, input_size, hidden_size, dropout=0.2):
                super(SimpleNN, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1)
                )
            
            def forward(self, x):
                return self.model(x)
        
        # Initialize model
        self.model = SimpleNN(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=self.config['hyperparameters']['dropout']
        ).to(self.device)
        
        # Split data into train/validation (80/20)
        indices = np.random.permutation(len(X))
        split_idx = int(0.8 * len(X))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X[train_indices]).to(self.device)
        y_train = torch.FloatTensor(y[train_indices]).view(-1, 1).to(self.device)
        X_val = torch.FloatTensor(X[val_indices]).to(self.device)
        y_val = torch.FloatTensor(y[val_indices]).view(-1, 1).to(self.device)
        
        # Check for NaN values in tensors - CRITICAL FIX
        if torch.isnan(X_train).any():
            print("WARNING: NaN values detected in X_train. Replacing with zeros.")
            X_train = torch.nan_to_num(X_train, nan=0.0)
        
        if torch.isnan(y_train).any():
            print("WARNING: NaN values detected in y_train. Replacing with mean.")
            y_mean = torch.mean(y_train[~torch.isnan(y_train)])
            y_train = torch.nan_to_num(y_train, nan=y_mean.item())
        
        if torch.isnan(X_val).any():
            print("WARNING: NaN values detected in X_val. Replacing with zeros.")
            X_val = torch.nan_to_num(X_val, nan=0.0)
        
        if torch.isnan(y_val).any():
            print("WARNING: NaN values detected in y_val. Replacing with mean.")
            y_mean = torch.mean(y_val[~torch.isnan(y_val)])
            y_val = torch.nan_to_num(y_val, nan=y_mean.item())
        
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['hyperparameters']['learning_rate'])
        batch_size = min(self.config['hyperparameters']['batch_size'], len(X_train))
        num_epochs = self.config['hyperparameters']['epochs']
        
        # Training loop with gradient clipping - CRITICAL FIX
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            # Process in batches
            for i in range(0, len(X_train), batch_size):
                # Get batch
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"WARNING: NaN loss detected at epoch {epoch+1}, batch {i//batch_size}. Skipping update.")
                    continue
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping - CRITICAL FIX
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss = train_loss / ((len(X_train) - 1) // batch_size + 1)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_losses.append(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Return training history
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'features': available_features
        }

    def train(self, train_data, val_data=None, sequence_length=7):
        """Train the sleep quality model"""
        # Preprocess data
        X_train, y_train, _, _, available_features = self.preprocess_data(train_data, sequence_length)
        
        if val_data is not None:
            X_val, y_val, _, _, _ = self.preprocess_data(val_data, sequence_length)
        else:
            # Use 20% of training data as validation
            split_idx = int(0.8 * len(X_train))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        # Store feature columns
        self.feature_columns = available_features
        
        # Initialize model
        input_size = len(available_features)
        hidden_size = self.config['hyperparameters']['hidden_size']
        num_layers = self.config['hyperparameters']['num_layers']
        dropout = self.config['hyperparameters']['dropout']
        
        self.model = SleepQualityLSTM(input_size, hidden_size, num_layers, dropout).to(self.device)
        
        # Training parameters
        learning_rate = self.config['hyperparameters']['learning_rate']
        batch_size = self.config['hyperparameters']['batch_size']
        num_epochs = self.config['hyperparameters']['epochs']
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            # Process in batches
            for i in range(0, len(X_train), batch_size):
                # Get batch
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            # Record losses
            train_loss = train_loss / (len(X_train) // batch_size)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Return training history
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'features': available_features
        }
    
    def predict(self, data, sequence_length=7):
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Validate input data
        validated_data = []
        for _, row in data.iterrows():
            try:
                # Extract required features
                input_dict = {col: row[col] for col in self.feature_columns if col in row}
                # Validate with ModelInput
                ModelInput(**input_dict)
                validated_data.append(row)
            except Exception as e:
                print(f"Invalid input data: {e}")
                # Skip invalid data
                continue
        
        # Create a DataFrame from validated data
        if not validated_data:
            raise ValueError("No valid input data after validation")
        
        valid_df = pd.DataFrame(validated_data)
        
        # Preprocess data
        X, _, user_ids, dates, _ = self.preprocess_data(valid_df, sequence_length)
        
        # Move to device
        X = X.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy().flatten()
        
        # Create results dataframe
        results = pd.DataFrame({
            'user_id': user_ids,
            'date': dates,
            'predicted_sleep_efficiency': predictions
        })
        
        # Validate output
        validated_outputs = []
        for _, row in results.iterrows():
            try:
                # Validate with ModelOutput
                output = ModelOutput(
                    predicted_sleep_efficiency=row['predicted_sleep_efficiency'],
                    prediction_confidence=0.8  # Default confidence
                )
                validated_outputs.append(output.dict())
            except Exception as e:
                print(f"Invalid output: {e}")
                # Clamp to valid range
                pred_eff = max(0.0, min(1.0, row['predicted_sleep_efficiency']))
                output = ModelOutput(
                    predicted_sleep_efficiency=pred_eff,
                    prediction_confidence=0.5  # Lower confidence for fixed outputs
                )
                validated_outputs.append(output.dict())
        
        # Convert validated outputs back to DataFrame
        return pd.DataFrame(validated_outputs)
    
    def calculate_sleep_score(self, sleep_efficiency, subjective_rating=None, additional_metrics=None):
        """Calculate an overall sleep score based on sleep efficiency and other metrics"""
        # Create sleep data dictionary from parameters
        sleep_data = {
            'sleep_efficiency': sleep_efficiency,
            'subjective_rating': subjective_rating
        }
        
        # Add additional metrics if available
        if additional_metrics is not None:
            sleep_data.update(additional_metrics)
        
        # Use the new calculator
        return self.sleep_score_calculator.calculate_score(sleep_data)
    
    def calculate_comprehensive_sleep_score(sleep_data, include_details=True):
        """
        Calculate a comprehensive sleep score using all available sleep data metrics
        including basic sleep metrics, demographics, and external factors.
        
        Args:
            sleep_data: Dict containing all sleep metrics and user attributes
            include_details: If True, return component scores along with total
            
        Returns:
            dict: Sleep score and component details
        """
        # Initialize component scores dictionary
        component_scores = {}
        
        # 1. Duration component (0-100)
        if 'sleep_duration_hours' in sleep_data:
            duration = sleep_data['sleep_duration_hours']
            # Ideal range: 7-9 hours
            if duration < 7:
                # Too little sleep - penalize more severely
                component_scores['duration'] = max(0, 70 * duration / 7)
            elif duration <= 9:
                # Ideal range - full score
                component_scores['duration'] = 100
            else:
                # Too much sleep - gentle penalty
                component_scores['duration'] = max(70, 100 - 15 * (duration - 9))
        
        # 2. Efficiency component (0-100)
        if 'sleep_efficiency' in sleep_data:
            efficiency = sleep_data['sleep_efficiency']
            # Ideal range: 0.85-0.95
            if efficiency < 0.85:
                # Below ideal - score decreases linearly
                component_scores['efficiency'] = 100 * efficiency / 0.85
            elif efficiency <= 0.95:
                # Ideal range - full score
                component_scores['efficiency'] = 100
            else:
                # Above ideal - small penalty
                component_scores['efficiency'] = max(80, 100 - 200 * (efficiency - 0.95))
        elif 'sleep_duration_hours' in sleep_data and 'time_in_bed_hours' in sleep_data:
            # Calculate if not provided directly
            efficiency = sleep_data['sleep_duration_hours'] / sleep_data['time_in_bed_hours']
            efficiency = max(0, min(1, efficiency))
            if efficiency < 0.85:
                component_scores['efficiency'] = 100 * efficiency / 0.85
            elif efficiency <= 0.95:
                component_scores['efficiency'] = 100
            else:
                component_scores['efficiency'] = max(80, 100 - 200 * (efficiency - 0.95))
        
        # 3. Sleep onset component (0-100)
        if 'sleep_onset_latency_minutes' in sleep_data:
            latency = sleep_data['sleep_onset_latency_minutes']
            if latency < 5:
                # Too quick - may indicate exhaustion
                component_scores['onset'] = 80
            elif latency <= 20:
                # Ideal range - full score
                component_scores['onset'] = 100
            elif latency <= 30:
                # Slightly delayed - mild penalty
                component_scores['onset'] = 90 - (latency - 20)
            elif latency <= 60:
                # Moderately delayed - steeper penalty
                component_scores['onset'] = 80 - (latency - 30) * 1.5
            else:
                # Severely delayed - largest penalty
                component_scores['onset'] = max(0, 35 - (latency - 60) * 0.35)
        elif 'bedtime' in sleep_data and 'sleep_onset_time' in sleep_data:
            # Calculate if not provided directly
            from datetime import datetime
            if isinstance(sleep_data['bedtime'], str):
                bedtime = datetime.strptime(sleep_data['bedtime'], '%Y-%m-%d %H:%M:%S')
            else:
                bedtime = sleep_data['bedtime']
                
            if isinstance(sleep_data['sleep_onset_time'], str):
                sleep_onset = datetime.strptime(sleep_data['sleep_onset_time'], '%Y-%m-%d %H:%M:%S')
            else:
                sleep_onset = sleep_data['sleep_onset_time']
            
            latency = (sleep_onset - bedtime).total_seconds() / 60
            if latency < 5:
                component_scores['onset'] = 80
            elif latency <= 20:
                component_scores['onset'] = 100
            elif latency <= 30:
                component_scores['onset'] = 90 - (latency - 20)
            elif latency <= 60:
                component_scores['onset'] = 80 - (latency - 30) * 1.5
            else:
                component_scores['onset'] = max(0, 35 - (latency - 60) * 0.35)
        
        # 4. Continuity component (awakenings and time awake) (0-100)
        awakenings_score = None
        if 'awakenings_count' in sleep_data:
            awakenings = sleep_data['awakenings_count']
            if awakenings <= 2:
                # Ideal range - full score
                awakenings_score = 100
            else:
                # Penalize each additional awakening
                awakenings_score = max(0, 100 - 10 * (awakenings - 2))
        
        awake_time_score = None
        if 'total_awake_minutes' in sleep_data:
            awake_time = sleep_data['total_awake_minutes']
            if awake_time <= 20:
                # Ideal range - full score
                awake_time_score = 100
            elif awake_time <= 45:
                # Mild disruption
                awake_time_score = 90 - (awake_time - 20) * 0.8
            else:
                # Severe disruption
                awake_time_score = max(0, 70 - (awake_time - 45) * 0.7)
        elif 'time_awake_minutes' in sleep_data:
            # Alternative field name
            awake_time = sleep_data['time_awake_minutes']
            if awake_time <= 20:
                awake_time_score = 100
            elif awake_time <= 45:
                awake_time_score = 90 - (awake_time - 20) * 0.8
            else:
                awake_time_score = max(0, 70 - (awake_time - 45) * 0.7)
        
        # Combine awakening scores if both are available
        if awakenings_score is not None and awake_time_score is not None:
            # Weight time awake more heavily than count
            component_scores['continuity'] = 0.4 * awakenings_score + 0.6 * awake_time_score
        elif awakenings_score is not None:
            component_scores['continuity'] = awakenings_score
        elif awake_time_score is not None:
            component_scores['continuity'] = awake_time_score
        
        # 5. Timing component (bedtime and waketime) (0-100)
        bedtime_score = None
        if 'bedtime' in sleep_data:
            # Extract hour from datetime
            if hasattr(sleep_data['bedtime'], 'hour'):
                bedtime_hour = sleep_data['bedtime'].hour
            else:
                # Try to parse from string
                try:
                    from datetime import datetime
                    dt = datetime.strptime(sleep_data['bedtime'], '%Y-%m-%d %H:%M:%S')
                    bedtime_hour = dt.hour
                except:
                    bedtime_hour = None
            
            if bedtime_hour is not None:
                # Handle after midnight (early hours are actually late)
                if bedtime_hour < 5:
                    bedtime_hour += 24
                
                # Ideal bedtime: 9pm-11pm (21-23)
                if bedtime_hour < 21:
                    # Too early
                    bedtime_score = 70 + 15 * (bedtime_hour / 21)
                elif bedtime_hour <= 23:
                    # Ideal range
                    bedtime_score = 100
                elif bedtime_hour <= 25:  # Up to 1 AM
                    # Slightly late
                    bedtime_score = 100 - 15 * (bedtime_hour - 23) / 2
                else:
                    # Very late
                    bedtime_score = max(40, 85 - 15 * (bedtime_hour - 25))
        
        waketime_score = None
        if 'wake_time' in sleep_data:
            # Extract hour from datetime
            if hasattr(sleep_data['wake_time'], 'hour'):
                waketime_hour = sleep_data['wake_time'].hour
            else:
                # Try to parse from string
                try:
                    from datetime import datetime
                    dt = datetime.strptime(sleep_data['wake_time'], '%Y-%m-%d %H:%M:%S')
                    waketime_hour = dt.hour
                except:
                    waketime_hour = None
            
            if waketime_hour is not None:
                # Ideal wake time: 6am-8am (6-8)
                if waketime_hour < 5:
                    # Very early
                    waketime_score = max(40, 70 * waketime_hour / 5)
                elif waketime_hour < 6:
                    # Slightly early
                    waketime_score = 80 + 20 * (waketime_hour - 5)
                elif waketime_hour <= 8:
                    # Ideal range
                    waketime_score = 100
                elif waketime_hour <= 10:
                    # Slightly late
                    waketime_score = 100 - 15 * (waketime_hour - 8) / 2
                else:
                    # Very late
                    waketime_score = max(40, 85 - 15 * (waketime_hour - 10))
        
        # Combine timing scores if both are available
        if bedtime_score is not None and waketime_score is not None:
            component_scores['timing'] = (bedtime_score + waketime_score) / 2
        elif bedtime_score is not None:
            component_scores['timing'] = bedtime_score
        elif waketime_score is not None:
            component_scores['timing'] = waketime_score
        
        # 6. Subjective component (0-100)
        if 'subjective_rating' in sleep_data:
            rating = sleep_data['subjective_rating']
            # Convert rating to 0-100 scale 
            if 1 <= rating <= 10:
                component_scores['subjective'] = (rating - 1) / 9 * 100
            elif 0 <= rating <= 100:
                component_scores['subjective'] = rating
        
        # Calculate base score using weighted components
        component_weights = {
            'duration': 0.20,  # Sleep duration
            'efficiency': 0.15,  # Sleep efficiency
            'onset': 0.15,      # Sleep onset latency
            'continuity': 0.20, # Sleep continuity
            'timing': 0.10,     # Sleep timing
            'subjective': 0.20  # Subjective rating
        }
        
        # Handle special case: no sleep
        if sleep_data.get('no_sleep', False):
            base_score = 0
        else:
            # Calculate weighted average with available components
            weighted_score = 0
            total_weight = 0
            
            for component, weight in component_weights.items():
                if component in component_scores:
                    weighted_score += component_scores[component] * weight
                    total_weight += weight
            
            # Normalize if not all components are available
            base_score = weighted_score / total_weight * 100 if total_weight > 0 else 50
        
        # Apply demographic adjustments
        adjustment = 0
        adjustment_reasons = []
        
        # Age-based adjustments
        if 'age' in sleep_data:
            age = sleep_data['age']
            if age < 25:
                # Young adults may need more sleep
                if 'sleep_duration_hours' in sleep_data and sleep_data['sleep_duration_hours'] < 7:
                    adjustment -= 2  # Penalize insufficient sleep more for young adults
                    adjustment_reasons.append("Young adults need more sleep")
            elif age > 65:
                # Elderly often have more fragmented sleep naturally
                if 'awakenings_count' in sleep_data and component_scores.get('continuity', 0) < 70:
                    adjustment += 3  # Less penalty for awakenings in elderly
                    adjustment_reasons.append("Older adults naturally have more awakenings")
        
        # Profession-based adjustments
        if 'profession_category' in sleep_data:
            profession = sleep_data['profession_category']
            
            if profession in ['healthcare', 'shift_worker']:
                # Healthcare workers and shift workers often have disrupted schedules
                if component_scores.get('timing', 0) < 70:
                    adjustment += 5  # Less penalty for unusual sleep timing
                    adjustment_reasons.append(f"{profession.title()} workers often have disrupted schedules")
                
                # Also adjust for sleep efficiency and continuity
                if component_scores.get('continuity', 0) < 70:
                    adjustment += 3  # Less penalty for fragmented sleep in shift workers
                    adjustment_reasons.append(f"{profession.title()} workers often experience fragmented sleep")
            
            elif profession == 'tech':
                # Tech workers often have high screen time and sedentary work
                if 'deep_sleep_percentage' in sleep_data and sleep_data['deep_sleep_percentage'] < 0.15:
                    adjustment -= 3  # Penalize low deep sleep more for tech workers
                    adjustment_reasons.append("Tech workers need adequate deep sleep to offset screen exposure")
                
                # Late-night work is common
                if component_scores.get('timing', 0) < 70 and 'bedtime' in sleep_data:
                    hour = sleep_data['bedtime'].hour if hasattr(sleep_data['bedtime'], 'hour') else 22
                    if hour > 23:  # Past 11 PM
                        adjustment += 2  # Less penalty for later bedtimes
                        adjustment_reasons.append("Tech workers often have later work hours")
            
            elif profession == 'service':
                # Service industry often has variable shifts
                if component_scores.get('timing', 0) < 70:
                    adjustment += 3  # Less penalty for variable timing
                    adjustment_reasons.append("Service industry often requires variable shifts")
            
            elif profession == 'education':
                # Educators often have early start times
                if 'wake_time' in sleep_data:
                    hour = sleep_data['wake_time'].hour if hasattr(sleep_data['wake_time'], 'hour') else 6
                    if hour < 6:  # Very early waketime
                        adjustment += 3  # Less penalty for early rising
                        adjustment_reasons.append("Educators often need to wake up very early")
            
            elif profession == 'office':
                # Office workers tend to be sedentary
                if 'sleep_duration_hours' in sleep_data and sleep_data['sleep_duration_hours'] < 7:
                    adjustment -= 2  # Greater penalty for insufficient sleep
                    adjustment_reasons.append("Sedentary office workers need adequate sleep for health")
            
            elif profession == 'creative':
                # Creative professionals often have variable schedules
                if component_scores.get('timing', 0) < 70:
                    adjustment += 3  # Less penalty for irregular timing
                    adjustment_reasons.append("Creative professionals often work with flexible schedules")
                
                # May have later bedtimes during creative periods
                if 'bedtime' in sleep_data:
                    hour = sleep_data['bedtime'].hour if hasattr(sleep_data['bedtime'], 'hour') else 22
                    if hour > 23:  # Past 11 PM
                        adjustment += 2  # Less penalty for later bedtimes
                        adjustment_reasons.append("Creative work often peaks in evening/night hours")
            
            elif profession == 'retired':
                # Retired individuals often have more flexibility but may have age-related sleep changes
                if component_scores.get('continuity', 0) < 70:
                    adjustment += 4  # Much less penalty for fragmented sleep
                    adjustment_reasons.append("Retired individuals commonly experience more fragmented sleep")
                
                # Earlier bedtimes are common and healthy
                if 'bedtime' in sleep_data:
                    hour = sleep_data['bedtime'].hour if hasattr(sleep_data['bedtime'], 'hour') else 22
                    if hour < 21:  # Before 9 PM
                        adjustment += 3  # Reward earlier bedtimes for retired people
                        adjustment_reasons.append("Earlier bedtimes are natural for retired individuals")
        

        # Apply the adjustment (limited to ±10 points)
        adjusted_score = base_score + max(-10, min(10, adjustment))
        
        # Ensure score is within valid range
        final_score = max(0, min(100, adjusted_score))
        
        if include_details:
            return {
                'total_score': int(round(final_score)),
                'base_score': int(round(base_score)),
                'component_scores': component_scores,
                'demographic_adjustment': adjustment,
                'adjustment_reasons': adjustment_reasons if adjustment != 0 else []
            }
        else:
            return int(round(final_score))
    
    def predict_with_confidence(self, data, sequence_length=7):
        """Make predictions with the trained model and include confidence scores"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess data
        X, _, user_ids, dates, _ = self.preprocess_data(data, sequence_length)
        
        # Move to device
        X = X.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy().flatten()
        
        # Calculate confidence scores
        confidences = []
        for i, (user_id, date) in enumerate(zip(user_ids, dates)):
            # Extract user-specific data
            user_data = data[data['user_id'] == user_id]
            
            # Base confidence level
            confidence = 0.7  # Start with a moderate confidence
            
            # Adjust based on data quantity
            data_quantity = len(user_data)
            if data_quantity >= 30:
                confidence += 0.1
            elif data_quantity < 10:
                confidence -= 0.1
            
            # Adjust based on data consistency
            user_data_sorted = user_data.sort_values('date')
            expected_dates = pd.date_range(start=user_data_sorted['date'].min(), end=user_data_sorted['date'].max())
            coverage_ratio = len(user_data) / len(expected_dates)
            
            if coverage_ratio >= 0.9:
                confidence += 0.05
            elif coverage_ratio < 0.7:
                confidence -= 0.05
            
            # Adjust based on prediction value (extreme values are less confident)
            prediction = predictions[i]
            if prediction < 0.3 or prediction > 0.95:
                confidence -= 0.05  # Less confident about extreme values
            
            # Ensure valid range [0.3, 0.95]
            confidence = max(0.3, min(0.95, confidence))
            confidences.append(round(confidence, 2))
        
        # Create results dataframe with confidence scores
        results = pd.DataFrame({
            'user_id': user_ids,
            'date': dates,
            'predicted_sleep_efficiency': predictions,
            'prediction_confidence': confidences
        })
        
        return results

    def calculate_sleep_score_with_confidence(self, sleep_efficiency, subjective_rating=None, additional_metrics=None):
        """Calculate an overall sleep score with confidence based on sleep efficiency and other metrics"""
        # Calculate the sleep score
        score = self.calculate_sleep_score(sleep_efficiency, subjective_rating, additional_metrics)
        
        # Base confidence
        confidence = 0.80  # Start with 80% confidence
        
        # Adjust based on input data completeness
        if subjective_rating is None:
            confidence -= 0.10  # Lower confidence without subjective input
        
        if additional_metrics:
            # Add confidence for each additional metric
            confidence += 0.02 * min(5, len(additional_metrics))  # Cap at +0.10
            
            # Specific metrics that increase confidence
            if 'deep_sleep_percentage' in additional_metrics and 'rem_sleep_percentage' in additional_metrics:
                confidence += 0.05  # Better sleep stage data improves confidence
        
        # Adjust for extreme scores
        if score < 20 or score > 90:
            confidence -= 0.05  # Less confident about extreme scores
        
        # Ensure valid range [0.4, 0.95]
        confidence = max(0.4, min(0.95, confidence))
        
        return {
            'score': score,
            'confidence': round(confidence, 2)
        }

    def save(self, filepath):
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), f"{filepath}.pt")
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'hyperparameters': self.config['hyperparameters'],
            'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load a trained model and metadata"""
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Extract parameters
        self.feature_columns = metadata['feature_columns']
        hyperparameters = metadata['hyperparameters']
        
        # Initialize model
        input_size = len(self.feature_columns)
        hidden_size = hyperparameters['hidden_size']
        num_layers = hyperparameters['num_layers']
        dropout = hyperparameters['dropout']
        
        self.model = SleepQualityLSTM(input_size, hidden_size, num_layers, dropout).to(self.device)
        
        # Load model state
        self.model.load_state_dict(torch.load(f"{filepath}.pt", map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded from {filepath}")
        
    def generate_model_card(self, filepath, performance_metrics=None, training_data_description=None):
        generate_model_card_with_samples(self, filepath, performance_metrics, training_data_description)