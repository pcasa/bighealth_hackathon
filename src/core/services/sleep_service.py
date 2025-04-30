# src/core/services/sleep_service.py
class SleepService:
    """Service layer for sleep-related operations"""
    
    def __init__(self, 
                repository, 
                recommendation_engine, 
                sleep_calculator):
        self.repository = repository
        self.recommendation_engine = recommendation_engine
        self.sleep_calculator = sleep_calculator
    
    async def log_sleep_entry(self, entry):
        """Process and log sleep entry"""
        # Save entry
        self.repository.save_sleep_entry(entry)
        
        # Get user data
        user_data = self.repository.get_user_data(entry.user_id)
        
        # Calculate sleep score
        sleep_score = self.sleep_calculator.calculate_score(entry)
        
        # Generate recommendation
        progress_data = self.recommendation_engine.analyze_progress(
            entry.user_id, user_data
        )
        recommendation = self.recommendation_engine.generate_recommendation(
            entry.user_id, progress_data
        )
        
        # Save recommendation
        self.repository.save_recommendation(
            entry.user_id, recommendation
        )
        
        return {
            "status": "success",
            "sleep_score": sleep_score,
            "recommendation": recommendation
        }