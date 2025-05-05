# src/core/services/sleep_service.py
from datetime import datetime
import pandas as pd


class SleepService:
    def __init__(self, repository, sleep_quality_model, recommendation_engine, preprocessor):
        self.repository = repository
        self.sleep_quality_model = sleep_quality_model
        self.recommendation_engine = recommendation_engine
        self.preprocessor = preprocessor
    
    async def log_sleep_entry(self, entry_dict):
        # Save entry
        self.repository.save_sleep_entry(entry_dict)
        
        # Get user profile info
        user_id = entry_dict['user_id']
        user_profile = self.repository.get_user_profile(user_id)
        
        # Get user sleep data
        user_data = self.repository.get_sleep_data(user_id)
        
        # Preprocess data
        processed_data = self.preprocessor.preprocess_sleep_data(user_data)
        
        # Calculate sleep score with the model
        row = processed_data[processed_data['date'] == pd.to_datetime(entry_dict['date'])].iloc[0]
        score_result = self.sleep_quality_model.calculate_sleep_score_with_confidence(
            row.get('sleep_efficiency', 0.8),
            row.get('subjective_rating', 7),
            self._get_metrics_from_row(row)
        )
        
        # Analyze progress with recommendation engine
        progress_data = self.recommendation_engine.analyze_progress(user_id, processed_data)
        
        # Generate recommendation with recommendation engine
        recommendation = self.recommendation_engine.generate_recommendation(
            user_id, 
            progress_data,
            user_profile.get('profession', ''),
            user_profile.get('region', '')
        )
        
        # Save recommendation
        self.repository.save_recommendation(user_id, recommendation)
        
        # Use model's prediction method for forecasting
        prediction_results = self.sleep_quality_model.predict_with_confidence(processed_data)
        
        return {
            "status": "success",
            "message": "Sleep entry logged successfully",
            "entry_date": entry_dict['date'],
            "sleep_score": score_result,
            "trend": progress_data.get('trend'),
            "consistency": progress_data.get('consistency'),
            "key_metrics": progress_data.get('key_metrics', {}),
            "recommendation": recommendation,
            "predictions": prediction_results
        }
    
    def _prepare_features_for_model(self, processed_data):
        """Prepares data with all features required by the sleep quality model"""
        # Clone the dataframe to avoid modifying the original
        df = processed_data.copy()
        
        # Required features for the model (expected 20 features)
        required_cols = [
            'sleep_duration_hours', 'sleep_efficiency', 'awakenings_count',
            'total_awake_minutes', 'deep_sleep_percentage', 'rem_sleep_percentage', 
            'sleep_onset_latency_minutes', 'heart_rate_variability', 'average_heart_rate',
            'age_normalized'
        ]
        
        # Add profession columns
        profession_cols = [
            'profession_healthcare', 'profession_tech', 'profession_service',
            'profession_education', 'profession_office', 'profession_other'
        ]
        
        # Add season columns
        season_cols = ['season_Winter', 'season_Spring', 'season_Summer', 'season_Fall']
        
        # Combine all feature columns
        all_features = required_cols + profession_cols + season_cols
        
        # Add profession category if missing
        if 'profession_category' not in df.columns and 'profession' in df.columns:
            df['profession_category'] = df['profession'].apply(self._categorize_profession)
        
        # Add profession one-hot encoding
        if any(col not in df.columns for col in profession_cols):
            # Initialize all profession columns to 0
            for col in profession_cols:
                df[col] = 0.0
                
            # Set the correct profession to 1
            if 'profession_category' in df.columns:
                for category in ['healthcare', 'tech', 'service', 'education', 'office', 'other']:
                    mask = df['profession_category'] == category
                    df.loc[mask, f'profession_{category}'] = 1.0
            else:
                # Default to 'other' if no profession information
                df['profession_other'] = 1.0
        
        # Add season information if missing
        if any(col not in df.columns for col in season_cols):
            # Extract month or use current month
            if 'date' in df.columns:
                df['month'] = pd.to_datetime(df['date']).dt.month
            else:
                df['month'] = datetime.now().month
                
            # Initialize all season columns to 0
            for col in season_cols:
                df[col] = 0.0
                
            # Set the correct season to 1 based on month
            df.loc[df['month'].isin([12, 1, 2]), 'season_Winter'] = 1.0
            df.loc[df['month'].isin([3, 4, 5]), 'season_Spring'] = 1.0
            df.loc[df['month'].isin([6, 7, 8]), 'season_Summer'] = 1.0
            df.loc[df['month'].isin([9, 10, 11]), 'season_Fall'] = 1.0
        
        # Add age_normalized if missing
        if 'age_normalized' not in df.columns:
            if 'age' in df.columns:
                df['age_normalized'] = df['age'] / 100.0
            else:
                df['age_normalized'] = 0.35  # Default normalized age
        
        # Fill in other missing columns with default values
        for col in all_features:
            if col not in df.columns:
                if col == 'sleep_efficiency':
                    df[col] = 0.85  # Default 85% efficiency
                elif col == 'sleep_duration_hours':
                    df[col] = 7.0  # Default 7 hours
                elif col in ['deep_sleep_percentage', 'rem_sleep_percentage']:
                    df[col] = 0.2  # Default 20% for each sleep stage
                elif col == 'heart_rate_variability':
                    df[col] = 50.0  # Default HRV
                elif col == 'average_heart_rate':
                    df[col] = 65.0  # Default heart rate
                else:
                    df[col] = 0.0  # Default other values to 0
                    
        # Ensure all columns have proper data types
        for col in df.columns:
            if col not in ['date', 'user_id'] and pd.api.types.is_object_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _categorize_profession(self, profession):
        """Categorize profession into predefined categories"""
        if not isinstance(profession, str):
            return 'other'
            
        profession = profession.lower()
        
        if any(term in profession for term in ['doctor', 'nurse', 'medical', 'health', 'physician', 'therapist']):
            return 'healthcare'
        elif any(term in profession for term in ['developer', 'programmer', 'engineer', 'tech', 'it', 'software']):
            return 'tech'
        elif any(term in profession for term in ['retail', 'service', 'cashier', 'server', 'hospitality']):
            return 'service'
        elif any(term in profession for term in ['teacher', 'professor', 'educator', 'academic', 'tutor']):
            return 'education'
        elif any(term in profession for term in ['manager', 'director', 'executive', 'admin', 'office']):
            return 'office'
        else:
            return 'other'
    
    async def analyze_sleep(self, user_id: str, days: int = 30):
        """Analyze sleep data for a user"""
        try:
            # Get user profile and sleep data
            user_profile = self.repository.get_user_profile(user_id)
            if not user_profile:
                return {"status": "error", "message": f"User with ID {user_id} not found"}
                
            sleep_data = self.repository.get_sleep_data(user_id, days)
            if len(sleep_data) == 0:
                return {"status": "error", "message": f"No sleep data found for user with ID {user_id}"}
            
            # Preprocess the data
            processed_data = self.preprocessor.preprocess_sleep_data(sleep_data)
            
            # Add profile information to each sleep record
            for key, value in user_profile.items():
                if key not in processed_data.columns and key != 'user_id':
                    processed_data[key] = value
                    
            # Prepare the data with all required features for the sleep quality model
            model_ready_data = self._prepare_features_for_model(processed_data)
            
            # Calculate sleep scores
            scores = []
            confidence_values = []
            
            for _, row in model_ready_data.iterrows():
                try:
                    # Calculate sleep score and confidence
                    score_result = self.sleep_quality_model.calculate_sleep_score_with_confidence(
                        row.get('sleep_efficiency', 0.85),
                        row.get('subjective_rating', 7),
                        {
                            'deep_sleep_percentage': row.get('deep_sleep_percentage', 0.2),
                            'rem_sleep_percentage': row.get('rem_sleep_percentage', 0.25),
                            'sleep_onset_latency_minutes': row.get('sleep_onset_latency_minutes', 15),
                            'awakenings_count': row.get('awakenings_count', 2),
                            'heart_rate_variability': row.get('heart_rate_variability', 50),
                            'average_heart_rate': row.get('average_heart_rate', 65)
                        }
                    )
                    
                    # Add date to score
                    date_value = row['date'] if 'date' in row else None
                    
                    scores.append({
                        'date': date_value,
                        'score': score_result['score'],
                        'confidence': score_result['confidence']
                    })
                    
                    confidence_values.append(score_result['confidence'])
                    
                except Exception as e:
                    print(f"Error calculating sleep score for entry: {str(e)}")
                    continue
            
            # Use the recommendation engine's analyze_progress method to get progress data
            # This avoids duplicating logic that already exists in the recommendation engine
            progress_data = self.recommendation_engine.analyze_progress(user_id, processed_data)
            
            # Make sure 'consistency' is included if the recommendation engine doesn't provide it
            if 'consistency' not in progress_data:
                # Check if user profile has sleep_consistency
                if 'sleep_consistency' in user_profile:
                    consistency_value = user_profile['sleep_consistency']
                    if consistency_value > 0.8:
                        progress_data['consistency'] = 'high'
                    elif consistency_value > 0.5:
                        progress_data['consistency'] = 'medium'
                    else:
                        progress_data['consistency'] = 'low'
                else:
                    # Default to medium if not available
                    progress_data['consistency'] = 'medium'
            
            # Generate recommendation using the progress data from the recommendation engine
            recommendation = self.recommendation_engine.generate_recommendation(
                user_id, 
                progress_data, 
                user_profile.get('profession', ''), 
                user_profile.get('region', '')
            )
            
            # Calculate average score and confidence
            avg_score = sum(entry['score'] for entry in scores) / len(scores) if scores else 0
            avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
            
            # Save recommendation
            self.repository.save_recommendation(user_id, recommendation)
            
            # Return analysis
            return {
                "user_id": user_id,
                "scores": scores,
                "average_score": avg_score,
                "recommendations": recommendation,
                "overall_confidence": avg_confidence
            }
            
        except Exception as e:
            print(f"Error analyzing sleep for user {user_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"Error analyzing sleep data: {str(e)}"}

    def _prepare_progress_data(self, processed_data, scores):
        """Prepares progress data with all required keys for the recommendation engine"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Start with an empty progress data dictionary
        progress_data = {
            'trend': 'stable',  # Default trend
            'sleep_quality': [],
            'sleep_efficiency': [],
            'sleep_duration': [],
            'sleep_onset_latency': [],
            'awakenings': [],
            'dates': []
        }
        
        # Ensure processed_data is not empty
        if processed_data.empty:
            return progress_data
        
        # Ensure date is in datetime format
        if 'date' in processed_data.columns:
            processed_data['date'] = pd.to_datetime(processed_data['date'])
            
        # Sort by date
        processed_data = processed_data.sort_values('date')
        
        # Fill in progress data from processed_data
        progress_data['dates'] = processed_data['date'].tolist()
        
        # Extract metrics from processed_data
        if 'sleep_efficiency' in processed_data.columns:
            progress_data['sleep_efficiency'] = processed_data['sleep_efficiency'].tolist()
        
        if 'sleep_duration_hours' in processed_data.columns:
            progress_data['sleep_duration'] = processed_data['sleep_duration_hours'].tolist()
        
        if 'sleep_onset_latency_minutes' in processed_data.columns:
            progress_data['sleep_onset_latency'] = processed_data['sleep_onset_latency_minutes'].tolist()
        
        if 'awakenings_count' in processed_data.columns:
            progress_data['awakenings'] = processed_data['awakenings_count'].tolist()
        
        # Add sleep quality from scores
        if scores:
            # Create a mapping of date to score
            score_map = {str(entry['date']): entry['score'] for entry in scores if entry['date'] is not None}
            
            # Fill sleep_quality list
            for date in progress_data['dates']:
                date_str = str(date)
                if date_str in score_map:
                    progress_data['sleep_quality'].append(score_map[date_str])
                else:
                    # If no score for this date, estimate from sleep_efficiency
                    idx = progress_data['dates'].index(date)
                    if idx < len(progress_data['sleep_efficiency']):
                        # Approximate score from sleep_efficiency (0-1 to 0-10 scale)
                        approx_score = progress_data['sleep_efficiency'][idx] * 10
                        progress_data['sleep_quality'].append(approx_score)
                    else:
                        # Default score if no efficiency data
                        progress_data['sleep_quality'].append(5.0)
        
        # Calculate trend based on sleep quality or efficiency
        if len(progress_data['sleep_quality']) >= 5:
            # Use last 5 entries to determine trend
            recent_quality = progress_data['sleep_quality'][-5:]
            
            # Calculate linear regression slope
            x = list(range(len(recent_quality)))
            y = recent_quality
            
            if len(x) > 1:  # Ensure we have at least 2 points for regression
                slope = np.polyfit(x, y, 1)[0]
                
                # Determine trend based on slope
                if slope > 0.2:  # Significant improvement
                    progress_data['trend'] = 'improving'
                elif slope < -0.2:  # Significant decline
                    progress_data['trend'] = 'declining'
                else:  # Stable
                    progress_data['trend'] = 'stable'
            else:
                progress_data['trend'] = 'stable'
        elif len(progress_data['sleep_efficiency']) >= 5:
            # Use efficiency if quality not available
            recent_efficiency = progress_data['sleep_efficiency'][-5:]
            
            x = list(range(len(recent_efficiency)))
            y = recent_efficiency
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                
                if slope > 0.02:  # Smaller thresholds for efficiency (0-1 scale)
                    progress_data['trend'] = 'improving'
                elif slope < -0.02:
                    progress_data['trend'] = 'declining'
                else:
                    progress_data['trend'] = 'stable'
            else:
                progress_data['trend'] = 'stable'
        
        # Add metrics averages
        metrics = ['sleep_quality', 'sleep_efficiency', 'sleep_duration', 'sleep_onset_latency', 'awakenings']
        for metric in metrics:
            if progress_data[metric]:
                avg_key = f'avg_{metric}'
                progress_data[avg_key] = sum(progress_data[metric]) / len(progress_data[metric])
            else:
                # Set default averages if data not available
                defaults = {
                    'sleep_quality': 5.0,
                    'sleep_efficiency': 0.85,
                    'sleep_duration': 7.0,
                    'sleep_onset_latency': 15.0,
                    'awakenings': 2.0
                }
                progress_data[f'avg_{metric}'] = defaults.get(metric, 0)
        
        return progress_data


    def _get_metrics_from_row(self, row):
        """Extract metrics from a row for the sleep quality model"""
        return {
            'deep_sleep_percentage': row.get('deep_sleep_percentage', 0.2),
            'rem_sleep_percentage': row.get('rem_sleep_percentage', 0.25),
            'sleep_onset_latency_minutes': row.get('sleep_onset_latency_minutes', 15),
            'awakenings_count': row.get('awakenings_count', 2),
            'total_awake_minutes': row.get('total_awake_minutes', 20)
        }