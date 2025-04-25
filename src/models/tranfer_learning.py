import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json
import yaml
from datetime import datetime

from src.models.sleep_quality import SleepQualityLSTM

class TransferLearning:
    def __init__(self, config_path='config/model_config.yaml'):
        """Initialize the transfer learning module"""
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.config = config['transfer_learning']
            self.model_config = config['sleep_quality_model']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = None
        self.user_models = {}
    
    def load_base_model(self, model_path):
        """Load the pre-trained base model"""
        # Load metadata
        with open(f"{model_path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Extract parameters
        self.feature_columns = metadata['feature_columns']
        hyperparameters = metadata['hyperparameters']
        
        # Initialize model
        input_size = len(self.feature_columns)
        hidden_size = hyperparameters['hidden_size']
        num_layers = hyperparameters['num_layers']
        dropout = hyperparameters['dropout']
        
        self.base_model = SleepQualityLSTM(input_size, hidden_size, num_layers, dropout).to(self.device)
        
        # Load model state
        self.base_model.load_state_dict(torch.load(f"{model_path}.pt", map_location=self.device))
        self.base_model.eval()
        
        print(f"Base model loaded from {model_path}")
    
    def adapt_to_user(self, user_id, user_data, sequence_length=7):
        """Adapt the base model to a specific user"""
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
        
        # Check if we have enough data for this user
        user_samples = user_data[user_data['user_id'] == user_id]
        if len(user_samples) < self.config['hyperparameters']['min_user_samples'] + sequence_length:
            print(f"Not enough data for user {user_id}. Need at least {self.config['hyperparameters']['min_user_samples'] + sequence_length} samples.")
            return None
        
        # Preprocess user data
        X, y = self._preprocess_user_data(user_samples, sequence_length)
        
        # Create a new model instance for this user (clone of base model)
        user_model = self._clone_model(self.base_model)
        
        # Freeze certain layers if specified
        if 'freeze_layers' in self.config['hyperparameters'] and self.config['hyperparameters']['freeze_layers']:
            self._freeze_layers(user_model, self.config['hyperparameters']['freeze_layers'])
        
        # Fine-tune the model on user data
        user_model, history = self._fine_tune_model(user_model, X, y)
        
        # Store the user model
        self.user_models[user_id] = user_model
        
        return history
    
    def _preprocess_user_data(self, user_data, sequence_length):
        """Preprocess data for a single user"""
        # Sort by date
        user_data = user_data.sort_values('date')
        
        # Extract features
        available_features = [col for col in self.feature_columns if col in user_data.columns]
        
        # Convert to numpy for easier slicing
        feature_data = user_data[available_features].values
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(feature_data) - sequence_length):
            seq = feature_data[i:i+sequence_length]
            target = user_data.iloc[i+sequence_length]['sleep_efficiency']
            
            sequences.append(seq)
            targets.append(target)
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(sequences)).to(self.device)
        y = torch.FloatTensor(np.array(targets)).view(-1, 1).to(self.device)
        
        return X, y
    
    def _clone_model(self, model):
        """Create a clone of the model with same architecture and weights"""
        # Get model architecture parameters
        input_size = model.lstm.input_size
        hidden_size = model.hidden_size
        num_layers = model.num_layers
        dropout = model.dropout.p
        
        # Create new instance
        cloned_model = SleepQualityLSTM(input_size, hidden_size, num_layers, dropout).to(self.device)
        
        # Copy weights
        cloned_model.load_state_dict(model.state_dict())
        
        return cloned_model
    
    def _freeze_layers(self, model, layers_to_freeze):
        """Freeze specified layers of the model"""
        for name, param in model.named_parameters():
            for layer_name in layers_to_freeze:
                if layer_name in name:
                    param.requires_grad = False
    
    def _fine_tune_model(self, model, X, y):
        """Fine-tune the model on user data"""
        # Training parameters
        learning_rate = self.config['hyperparameters']['learning_rate']
        num_epochs = self.config['hyperparameters']['fine_tuning_epochs']
        regularization_weight = self.config['hyperparameters']['regularization_weight']
        
        # Split data into train and validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=regularization_weight
        )
        
        # Training loop
        model.train()
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            # Switch back to train mode
            model.train()
            
            # Record losses
            train_loss = loss.item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Set to evaluation mode
        model.eval()
        
        return model, {'train_losses': train_losses, 'val_losses': val_losses}
    
    def predict_for_user(self, user_id, user_data, sequence_length=7):
        """Make predictions for a specific user using their adapted model"""
        # Check if we have an adapted model for this user
        if user_id in self.user_models:
            model = self.user_models[user_id]
        else:
            # Fall back to base model
            print(f"No adapted model found for user {user_id}. Using base model.")
            if self.base_model is None:
                raise ValueError("Base model not loaded. Call load_base_model first.")
            model = self.base_model
        
        # Preprocess user data
        user_samples = user_data[user_data['user_id'] == user_id]
        
        # Sort by date
        user_samples = user_samples.sort_values('date')
        
        # Extract features
        available_features = [col for col in self.feature_columns if col in user_samples.columns]
        
        # Need at least sequence_length samples to make a prediction
        if len(user_samples) < sequence_length:
            print(f"Not enough data for user {user_id}. Need at least {sequence_length} samples.")
            return None
        
        # Convert to numpy for easier slicing
        feature_data = user_samples[available_features].values
        
        # Create sequence (use the most recent data)
        sequence = feature_data[-sequence_length:]
        
        # Convert to tensor
        X = torch.FloatTensor(np.array([sequence])).to(self.device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(X).cpu().numpy().flatten()[0]
        
        return prediction
    
    def save_user_model(self, user_id, filepath):
        """Save an adapted user model"""
        if user_id not in self.user_models:
            raise ValueError(f"No adapted model found for user {user_id}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state
        torch.save(self.user_models[user_id].state_dict(), f"{filepath}_{user_id}.pt")
        
        # Save metadata
        metadata = {
            'user_id': user_id,
            'feature_columns': self.feature_columns,
            'hyperparameters': self.model_config['hyperparameters'],
            'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'adaptation_technique': self.config['adaptation_technique'],
            'adaptation_hyperparameters': self.config['hyperparameters']
        }
        
        with open(f"{filepath}_{user_id}_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        print(f"User model saved to {filepath}_{user_id}")
    
    def load_user_model(self, user_id, filepath):
        """Load an adapted user model"""
        # Load metadata
        with open(f"{filepath}_{user_id}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Extract parameters
        self.feature_columns = metadata['feature_columns']
        hyperparameters = metadata['hyperparameters']
        
        # Initialize model
        input_size = len(self.feature_columns)
        hidden_size = hyperparameters['hidden_size']
        num_layers = hyperparameters['num_layers']
        dropout = hyperparameters['dropout']
        
        model = SleepQualityLSTM(input_size, hidden_size, num_layers, dropout).to(self.device)
        
        # Load model state
        model.load_state_dict(torch.load(f"{filepath}_{user_id}.pt", map_location=self.device))
        model.eval()
        
        # Store in user_models
        self.user_models[user_id] = model
        
        print(f"User model loaded from {filepath}_{user_id}")