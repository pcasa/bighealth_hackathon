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
from src.models.improved_sleep_score import ImprovedSleepScoreCalculator



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

class SleepQualityModel:
    def __init__(self, config_path='config/model_config.yaml'):
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
        
        # Add profession one-hot encoding if available
        if 'profession_category' in data.columns:
            prof_categories = ['healthcare', 'tech', 'service', 'education', 'office', 'other']
            for category in prof_categories:
                data[f'profession_{category}'] = (data['profession_category'] == category).astype(float)
        
        # Ensure features are in the right format
        required_features = self.config['features']
        available_features = [col for col in required_features if col in data.columns]
        
        # Add demographic and seasonal features if available
        demographic_features = [col for col in data.columns if (
            col.startswith('season_') or 
            col.startswith('profession_') or 
            col == 'age_normalized'
        )]
        
        available_features.extend([f for f in demographic_features if f not in available_features])
        
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
                data[missing_feature] = 0.0
                available_features.append(missing_feature)
        
        # Store feature columns
        self.feature_columns = available_features
        
        # Extract features and target
        X = data[available_features].values
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
        
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['hyperparameters']['learning_rate'])
        batch_size = min(self.config['hyperparameters']['batch_size'], len(X_train))
        num_epochs = self.config['hyperparameters']['epochs']
        
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
        
        # Preprocess data
        X, _, user_ids, dates, _ = self.preprocess_data(data, sequence_length)
        
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
        
        return results
    
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
    
    def calculate_comprehensive_sleep_score(self, sleep_data, include_details=True):
        """
        Calculate a comprehensive sleep score using all available sleep data metrics
        including age, profession, and seasonal context.
        
        Args:
            sleep_data: Dict containing sleep metrics and user attributes
            include_details: If True, return component scores along with total
            
        Returns:
            dict: Sleep score and component details
        """
        # Get basic sleep score from the calculator
        basic_score = self.sleep_score_calculator.calculate_score(sleep_data, include_details=True)
        
        # Extract components and base score
        component_scores = basic_score['component_scores']
        base_score = basic_score['total_score']
        
        # Apply demographic adjustments
        adjustment = 0
        
        # Age-based adjustments
        if 'age' in sleep_data:
            age = sleep_data['age']
            if age < 25:
                # Young adults may need more sleep
                if 'sleep_duration_hours' in sleep_data and sleep_data['sleep_duration_hours'] < 7:
                    adjustment -= 2  # Penalize insufficient sleep more for young adults
            elif age > 65:
                # Elderly often have more fragmented sleep naturally
                if 'awakenings_count' in sleep_data and component_scores.get('continuity', 0) < 70:
                    adjustment += 3  # Less penalty for awakenings in elderly
                
                # Elderly often go to bed earlier
                if 'bedtime' in sleep_data:
                    hour = sleep_data['bedtime'].hour if hasattr(sleep_data['bedtime'], 'hour') else 22
                    if hour < 21:  # Before 9 PM
                        adjustment += 2  # Earlier bedtime can be appropriate for elderly
        
        # Profession-based adjustments
        if 'profession_category' in sleep_data:
            profession = sleep_data['profession_category']
            
            if profession == 'healthcare' or profession == 'shift_worker':
                # Healthcare workers and shift workers often have disrupted schedules
                if 'sleep_pattern' in sleep_data and sleep_data['sleep_pattern'] == 'shift_worker':
                    if component_scores.get('timing', 0) < 70:
                        adjustment += 5  # Less penalty for unusual sleep timing
            
            elif profession in ['tech', 'office']:
                # Office/tech workers often have sedentary jobs with high screen time
                if 'deep_sleep_percentage' in sleep_data and sleep_data['deep_sleep_percentage'] < 0.15:
                    adjustment -= 3  # Penalize low deep sleep more for sedentary workers
        
        # Season-based adjustments
        if 'season' in sleep_data or 'month' in sleep_data:
            # Determine season if not directly provided
            season = sleep_data.get('season')
            if not season and 'month' in sleep_data:
                month = sleep_data['month']
                if isinstance(month, int) and 1 <= month <= 12:
                    if month in [12, 1, 2]:
                        season = 'Winter'
                    elif month in [3, 4, 5]:
                        season = 'Spring'
                    elif month in [6, 7, 8]:
                        season = 'Summer'
                    else:
                        season = 'Fall'
            
            if season:
                if season == 'Winter':
                    # Winter: people naturally sleep longer
                    if 'sleep_duration_hours' in sleep_data:
                        if sleep_data['sleep_duration_hours'] > 8:
                            adjustment += 1  # Natural to sleep longer in winter
                elif season == 'Summer':
                    # Summer: more daylight can affect sleep
                    if 'sleep_onset_latency_minutes' in sleep_data and sleep_data['sleep_onset_latency_minutes'] > 20:
                        adjustment += 2  # More tolerance for longer sleep onset in summer
        
        # Apply the adjustment (limited to ±10 points)
        adjusted_score = base_score + max(-10, min(10, adjustment))
        
        # Ensure score is within valid range
        final_score = max(0, min(100, adjusted_score))
        
        # Add adjustment explanation to results
        result = {
            'total_score': final_score,
            'base_score': base_score,
            'component_scores': component_scores,
            'demographic_adjustment': adjustment
        }
        
        # Add explanation of adjustments
        if adjustment != 0:
            adjustment_reasons = []
            
            if 'age' in sleep_data:
                if age < 25 and 'sleep_duration_hours' in sleep_data and sleep_data['sleep_duration_hours'] < 7:
                    adjustment_reasons.append("Young adults need more sleep")
                elif age > 65:
                    if 'awakenings_count' in sleep_data and component_scores.get('continuity', 0) < 70:
                        adjustment_reasons.append("Older adults naturally have more awakenings")
                    if 'bedtime' in sleep_data:
                        hour = sleep_data['bedtime'].hour if hasattr(sleep_data['bedtime'], 'hour') else 22
                        if hour < 21:
                            adjustment_reasons.append("Earlier bedtime appropriate for older adults")
            
            if 'profession_category' in sleep_data:
                if profession in ['healthcare', 'shift_worker'] and component_scores.get('timing', 0) < 70:
                    adjustment_reasons.append(f"{profession.title()} workers often have disrupted schedules")
                elif profession in ['tech', 'office'] and 'deep_sleep_percentage' in sleep_data and sleep_data['deep_sleep_percentage'] < 0.15:
                    adjustment_reasons.append("Office workers need more deep sleep to offset sedentary work")
            
            if season:
                if season == 'Winter' and 'sleep_duration_hours' in sleep_data and sleep_data['sleep_duration_hours'] > 8:
                    adjustment_reasons.append("Longer sleep in winter is natural")
                elif season == 'Summer' and 'sleep_onset_latency_minutes' in sleep_data and sleep_data['sleep_onset_latency_minutes'] > 20:
                    adjustment_reasons.append("Longer daylight in summer can affect sleep onset")
            
            result['adjustment_reasons'] = adjustment_reasons
        
        return result
    
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
        """Generate a model card for the sleep quality model"""
        if self.model is None:
            raise ValueError("No trained model available")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Model card content
        model_card = {
            "model_name": "Sleep Quality Prediction Model",
            "version": "1.0",
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "model_type": "LSTM Neural Network",
            "purpose": "Predict sleep efficiency and calculate sleep quality scores",
            "features": self.feature_columns,
            "hyperparameters": self.config['hyperparameters'],
            "architecture": {
                "input_size": len(self.feature_columns),
                "hidden_size": self.config['hyperparameters']['hidden_size'],
                "num_layers": self.config['hyperparameters']['num_layers'],
                "dropout": self.config['hyperparameters']['dropout']
            },
            "performance_metrics": performance_metrics or {},
            "training_data": training_data_description or "Not specified",
            "intended_use": "Analyzing sleep patterns and providing personalized sleep quality scores",
            "limitations": [
                "Requires at least 7 days of consecutive sleep data",
                "May not be accurate for users with highly irregular sleep patterns",
                "Not clinically validated for sleep disorder diagnosis"
            ],
            "ethical_considerations": [
                "Should not be used as the sole basis for medical decisions",
                "Privacy considerations for handling sensitive sleep data"
            ],
            "maintenance": {
                "recommended_retraining_frequency": "Every 3 months with new data",
                "data_drift_monitoring": "Implemented to detect changes in feature distributions"
            }
        }
        
        # Save model card
        with open(filepath, 'w') as f:
            json.dump(model_card, f, indent=2)
        
        # Also create a markdown version
        md_filepath = filepath.replace('.json', '.md')
        with open(md_filepath, 'w') as f:
            f.write(f"# {model_card['model_name']} v{model_card['version']}\n\n")
            f.write(f"**Created:** {model_card['created_date']}\n\n")
            
            f.write("## Purpose\n")
            f.write(f"{model_card['purpose']}\n\n")
            
            f.write("## Model Type\n")
            f.write(f"{model_card['model_type']}\n\n")
            
            f.write("## Features\n")
            for feature in model_card['features']:
                f.write(f"- {feature}\n")
            f.write("\n")
            
            f.write("## Architecture\n")
            for key, value in model_card['architecture'].items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            if performance_metrics:
                f.write("## Performance Metrics\n")
                for metric, value in model_card['performance_metrics'].items():
                    f.write(f"- {metric}: {value}\n")
                f.write("\n")
            
            f.write("## Limitations\n")
            for limitation in model_card['limitations']:
                f.write(f"- {limitation}\n")
            f.write("\n")
            
            f.write("## Ethical Considerations\n")
            for consideration in model_card['ethical_considerations']:
                f.write(f"- {consideration}\n")
            f.write("\n")
            
            f.write("## Maintenance\n")
            for key, value in model_card['maintenance'].items():
                f.write(f"- {key}: {value}\n")
        
        print(f"Model card saved to {filepath} and {md_filepath}")