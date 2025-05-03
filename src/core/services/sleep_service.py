# src/core/services/sleep_service.py
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
    
    async def analyze_sleep(self, user_id, days=30):
        """Analyze sleep data for a user using existing model and engine methods"""
        # Get user sleep data
        user_data = self.repository.get_sleep_data(user_id, days)
        
        # Check if we have enough data
        if len(user_data) < 3:
            return {
                "status": "error",
                "message": "Not enough sleep data for analysis"
            }
        
        # Get user profile
        user_profile = self.repository.get_user_profile(user_id)
        profession = user_profile.get('profession', '') if user_profile else ''
        region = user_profile.get('region', '') if user_profile else ''
        
        # Preprocess data
        user_data['date'] = pd.to_datetime(user_data['date'])
        processed_data = self.preprocessor.preprocess_sleep_data(user_data)
        
        # Analyze progress with recommendation engine
        progress_data = self.recommendation_engine.analyze_progress(user_id, processed_data)
        
        # Generate recommendation using the engine
        recommendation = self.recommendation_engine.generate_recommendation(
            user_id, progress_data, profession, region
        )
        
        # Calculate confidence using model's method
        prediction_confidence = self.sleep_quality_model.calculate_prediction_confidence(
            processed_data, progress_data
        )
        
        # Save recommendation
        self.repository.save_recommendation(user_id, recommendation)
        
        return {
            "trend": progress_data.get('trend'),
            "consistency": progress_data.get('consistency'),
            "improvement_rate": progress_data.get('improvement_rate'),
            "key_metrics": progress_data.get('key_metrics', {}),
            "recommendations": {
                "text": recommendation,
                "confidence": prediction_confidence
            },
            "overall_confidence": prediction_confidence
        }
    
    def _get_metrics_from_row(self, row):
        """Extract metrics from a row for the sleep quality model"""
        return {
            'deep_sleep_percentage': row.get('deep_sleep_percentage', 0.2),
            'rem_sleep_percentage': row.get('rem_sleep_percentage', 0.25),
            'sleep_onset_latency_minutes': row.get('sleep_onset_latency_minutes', 15),
            'awakenings_count': row.get('awakenings_count', 2),
            'total_awake_minutes': row.get('total_awake_minutes', 20)
        }