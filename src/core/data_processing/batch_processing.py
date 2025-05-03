import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging
from multiprocessing import Pool, cpu_count

from pydantic import Field
from pyparsing import Dict, Optional

from src.core.models.base_model import BaseModel
from src.core.data_processing.preprocessing import DataPreprocessor
from src.core.data_processing.feature_engineering import FeatureEngineering

class BatchConfig(BaseModel):
    """Configuration for batch processing"""
    batch_size: int = Field(1000, ge=100, le=10000)
    checkpoint_dir: str = "data/checkpoints"
    output_dir: str = "data/processed"
    parallel_processing: bool = True
    max_workers: int = Field(4, ge=1, le=16)


class BatchProgressTracker(BaseModel):
    """Track progress of a batch processing run"""
    started_at: datetime
    last_processed_row: int = 0
    total_rows: int = 0
    processed_batches: int = 0
    error_count: int = 0
    success_count: int = 0
    status: str = "running"  # running, completed, failed
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        if self.total_rows == 0:
            return 0.0
        return (self.last_processed_row / self.total_rows) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds"""
        return (datetime.now() - self.started_at).total_seconds()



class BatchProcessor:
    def __init__(self, batch_size=1000, checkpoint_dir='data/checkpoints', config: Optional[Dict] = None):
        """Initialize the batch processor"""
        self.config = BatchConfig(**(config or {}))
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.preprocessor = DataPreprocessor()
        self.feature_engineering = FeatureEngineering()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/batch_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BatchProcessor')
        
        # Ensure checkpoint directory exists
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def process_data(self, input_file, output_file, parallel=True):
        """Process data in batches with checkpointing"""
        # Set output file if not provided
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(self.config.output_dir, f"{base_name}_processed.csv")
        
         # Set parallel processing flag
        parallel = parallel if parallel is not None else self.config.parallel_processing
        
        # Create progress tracker
        progress = BatchProgressTracker(
            started_at=datetime.now(),
            total_rows=0,  # Will be updated after reading file
            status="running"
        )
        
        self.logger.info(f"Starting batch processing of {input_file}")
        
        # Check for existing checkpoint
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{os.path.basename(input_file)}.checkpoint")
        start_row = 0
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_row = checkpoint['last_processed_row'] + 1
                self.logger.info(f"Resuming from checkpoint: row {start_row}")
        
        # Determine reader parameters based on file type
        file_extension = os.path.splitext(input_file)[1].lower()
        
        if file_extension == '.csv':
            reader_func = pd.read_csv
            reader_kwargs = {'chunksize': self.batch_size}
        elif file_extension in ['.parquet', '.pq']:
            # For parquet, we need to read the whole file then process in chunks
            data = pd.read_parquet(input_file)
            total_rows = len(data)
            batch_count = (total_rows - start_row) // self.batch_size + (1 if (total_rows - start_row) % self.batch_size > 0 else 0)
            self.logger.info(f"Processing {total_rows} rows in {batch_count} batches")
            
            result = self._process_dataframe_in_batches(data, start_row, output_file, parallel)
            return result
        else:
            self.logger.error(f"Unsupported file format: {file_extension}")
            return False
        
        # Process CSV data in chunks
        try:
            first_batch = True
            for i, chunk in enumerate(reader_func(input_file, **reader_kwargs)):
                # Skip to the checkpoint position
                if i * self.batch_size < start_row:
                    continue
                
                self.logger.info(f"Processing batch {i}, rows {i * self.batch_size}-{(i+1) * self.batch_size}")
                
                # Process batch
                processed_batch = self._process_batch(chunk, parallel)
                
                # Write batch to output
                if first_batch:
                    # First batch - create new file with headers
                    processed_batch.to_csv(output_file, index=False)
                    first_batch = False
                else:
                    # Append to existing file without headers
                    processed_batch.to_csv(output_file, mode='a', header=False, index=False)
                
                # Update checkpoint
                with open(checkpoint_file, 'w') as f:
                    json.dump({'last_processed_row': (i+1) * self.batch_size - 1}, f)
            
            # Processing completed successfully
            self.logger.info(f"Batch processing completed. Output saved to {output_file}")
            
            # Remove checkpoint file since processing is complete
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error during batch processing: {str(e)}")
            return False
    
    def _process_dataframe_in_batches(self, data, start_row, output_file, parallel):
        """Process a dataframe in batches"""
        total_rows = len(data)
        first_batch = True
        
        for batch_start in range(start_row, total_rows, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_rows)
            self.logger.info(f"Processing batch rows {batch_start}-{batch_end}")
            
            # Extract batch
            batch = data.iloc[batch_start:batch_end].copy()
            
            # Process batch
            processed_batch = self._process_batch(batch, parallel)
            
            # Write batch to output
            if first_batch:
                # First batch - create new file with headers
                processed_batch.to_csv(output_file, index=False)
                first_batch = False
            else:
                # Append to existing file without headers
                processed_batch.to_csv(output_file, mode='a', header=False, index=False)
            
            # Update checkpoint
            checkpoint_file = os.path.join(self.checkpoint_dir, f"{os.path.basename(output_file)}.checkpoint")
            with open(checkpoint_file, 'w') as f:
                json.dump({'last_processed_row': batch_end}, f)
        
        # Processing completed successfully
        self.logger.info(f"Batch processing completed. Output saved to {output_file}")
        
        # Remove checkpoint file since processing is complete
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{os.path.basename(output_file)}.checkpoint")
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            
        return True
    
    def _process_batch(self, batch, parallel=True):
        """Process a single batch of data"""
        if parallel and len(batch) > 100:  # Only use parallel for larger batches
            # Split batch into mini-batches for parallel processing
            num_processes = min(cpu_count(), 4)  # Limit to 4 processes max
            mini_batch_size = max(1, len(batch) // num_processes)
            mini_batches = []
            
            for i in range(0, len(batch), mini_batch_size):
                mini_batches.append(batch.iloc[i:i+mini_batch_size])
            
            # Process mini-batches in parallel
            with Pool(num_processes) as pool:
                processed_mini_batches = pool.map(self._process_mini_batch, mini_batches)
            
            # Combine results
            processed_batch = pd.concat(processed_mini_batches)
            
        else:
            # Process batch sequentially
            processed_batch = self._process_mini_batch(batch)
        
        return processed_batch
    
    def _process_mini_batch(self, mini_batch):
        """Process a mini-batch for parallel processing"""
        # Preprocess data
        preprocessed_data = self.preprocessor.preprocess_sleep_data(mini_batch)
        
        # Feature engineering
        feature_data = self.feature_engineering.create_features(preprocessed_data)
        
        return feature_data
    
    def load_and_process_file(self, input_file, process_func=None):
        """Load and process a file in batches"""
        file_extension = os.path.splitext(input_file)[1].lower()
        
        if file_extension == '.csv':
            reader_func = pd.read_csv
            reader_kwargs = {'chunksize': self.batch_size}
            
            all_processed = []
            
            for chunk in reader_func(input_file, **reader_kwargs):
                # Apply custom processing function if provided
                if process_func:
                    processed_chunk = process_func(chunk)
                else:
                    processed_chunk = chunk
                
                all_processed.append(processed_chunk)
            
            # Combine all chunks
            return pd.concat(all_processed)
            
        elif file_extension in ['.parquet', '.pq']:
            # For parquet, we need to read the whole file
            data = pd.read_parquet(input_file)
            
            # Apply custom processing function if provided
            if process_func:
                return process_func(data)
            else:
                return data
        else:
            self.logger.error(f"Unsupported file format: {file_extension}")
            return None