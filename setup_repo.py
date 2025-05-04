#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sleep Insights App Setup Script

This script prepares the Sleep Insights app for development by:
1. Generating training data
2. Training models
3. Running the enhanced demo
4. Analyzing the demo data
5. Running enhanced analytics

Usage:
    python setup.py [--skip-training] [--skip-demo] [--skip-analysis] [--start-api]

Options:
    --skip-training     Skip the model training step (use if you already have models)
    --skip-demo         Skip running the enhanced demo
    --skip-analysis     Skip the analysis steps
    --start-api         Start the API server after setup
"""

import os
import sys
import time
import argparse
import subprocess
import logging
from pathlib import Path
import random
import csv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/setup.log')
    ]
)

# ANSI color codes for better readability
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[0;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def print_color(color, text):
    """Print colored text"""
    print(f"{color}{text}{Colors.NC}")

def run_command(command, description, exit_on_error=True):
    """Run a shell command with error handling"""
    print_color(Colors.YELLOW, f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print_color(Colors.GREEN, f"✓ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print_color(Colors.RED, f"✗ {description} failed with code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        if exit_on_error:
            sys.exit(1)
        return None

def create_directories():
    """Create necessary directories"""
    print_color(Colors.YELLOW, "Creating required directories...")
    directories = [
        "data/raw",
        "data/processed",
        "data/enhanced_demo",
        "data/logs",
        "models",
        "reports/enhanced_analysis",
        "reports/user_insights",
        "logs"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print_color(Colors.GREEN, "✓ Directories created")

def install_dependencies():
    """Install required dependencies"""
    if Path("requirements.txt").exists():
        print_color(Colors.YELLOW, "Installing dependencies...")
        subprocess.run(["uv", "sync"], check=True)
        subprocess.run(["uv", "pip", "install", "torch"], check=True)
        
        # Check if torch is installed
        try:
            import pkg_resources
            pkg_resources.get_distribution("torch")
            print_color(Colors.GREEN, "✓ PyTorch is installed")
        except pkg_resources.DistributionNotFound:
            print_color(Colors.YELLOW, "Installing PyTorch separately...")
            subprocess.run(["pip", "install", "torch"], check=True)
    else:
        print_color(Colors.RED, "requirements.txt not found. Please ensure all dependencies are installed.")

def generate_training_data():
    """Step 1: Generate training data"""
    print_color(Colors.BLUE, "\n=== Step 1: Generating Training Data ===")
    run_command(
        "uv run scripts/data_generation_script.py", 
        "Data generation"
    )

def train_models():
    """Step 2: Train models"""
    print_color(Colors.BLUE, "\n=== Step 2: Training Models ===")
    run_command(
        "uv run scripts/train_models.py --data-dir data/raw --output-dir models", 
        "Model training", 
        exit_on_error=False  # Don't exit if model training fails, use fallbacks
    )

def run_enhanced_demo():
    """Step 3: Run enhanced demo"""
    print_color(Colors.BLUE, "\n=== Step 3: Running Enhanced Demo ===")
    run_command(
        "uv run scripts/run_enhanced_demo.py --output-dir data/enhanced_demo --user-count 200 --wearable-percentage 30", 
        "Enhanced demo generation"
    )

def analyze_demo_data():
    """Step 4: Analyze demo data"""
    print_color(Colors.BLUE, "\n=== Step 4: Analyzing Demo Data ===")
    run_command(
        "uv run scripts/sleep_advisor.py --form-data-file data/enhanced_demo/data/sleep_data.csv --historical-data data/enhanced_demo/data/sleep_data.csv --output-dir data/enhanced_demo/recommendations", 
        "Sleep advisor analysis", 
        exit_on_error=False
    )

def run_enhanced_analytics():
    """Step 5: Run enhanced analytics"""
    print_color(Colors.BLUE, "\n=== Step 5: Running Enhanced Analytics ===")
    run_command(
        "uv run scripts/analyze_enhanced_data.py --data-dir data/enhanced_demo/data --output-dir reports/enhanced_analysis", 
        "Enhanced analytics", 
        exit_on_error=False
    )

def generate_user_insights():
    """Step 6: Generate user insights"""
    print_color(Colors.BLUE, "\n=== Step 6: Generating User Insights ===")
    
    # Pick a random user ID from the generated data
    sample_user_id = None
    
    try:
        if os.path.exists('data/enhanced_demo/data/users.csv'):
            with open('data/enhanced_demo/data/users.csv', 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = list(reader)
                if rows:
                    sample_user_id = random.choice(rows)[0]
    except Exception as e:
        print_color(Colors.RED, f"Error picking sample user: {e}")
    
    if sample_user_id:
        print_color(Colors.YELLOW, f"Selected user ID: {sample_user_id}")
        run_command(
            f"uv run scripts/user-insights-generator.py --user-id {sample_user_id} --data-dir data/enhanced_demo/data --output-dir reports/user_insights", 
            "User insights generation", 
            exit_on_error=False
        )
    else:
        print_color(Colors.RED, "No sample user found. Skipping user insights generation.")

def start_api_server():
    """Start the API server"""
    print_color(Colors.BLUE, "\n=== Starting API Server ===")
    print_color(Colors.YELLOW, "API will be available at http://localhost:8000")
    print_color(Colors.YELLOW, "To stop the server, press Ctrl+C")
    subprocess.run(["uv", "run", "uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Sleep Insights App Setup Script')
    parser.add_argument('--skip-training', action='store_true', help='Skip the model training step')
    parser.add_argument('--skip-demo', action='store_true', help='Skip running the enhanced demo')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip the analysis steps')
    parser.add_argument('--start-api', action='store_true', help='Start the API server after setup')
    args = parser.parse_args()

    # Header
    print_color(Colors.BLUE, "================================================")
    print_color(Colors.BLUE, "Sleep Insights App - Development Setup Script")
    print_color(Colors.BLUE, "================================================")

    # Create directories and install dependencies
    create_directories()
    install_dependencies()
    
    # Step 1: Generate training data
    generate_training_data()
    
    # Step 2: Train models (optional)
    if not args.skip_training:
        train_models()
    else:
        print_color(Colors.YELLOW, "\nSkipping model training as requested")
    
    # Step 3: Run enhanced demo (optional)
    if not args.skip_demo:
        run_enhanced_demo()
    else:
        print_color(Colors.YELLOW, "\nSkipping enhanced demo as requested")
    
    # Steps 4-5: Analysis steps (optional)
    if not args.skip_analysis:
        analyze_demo_data()
        run_enhanced_analytics()
        generate_user_insights()
    else:
        print_color(Colors.YELLOW, "\nSkipping analysis steps as requested")
    
    # Summary
    print_color(Colors.GREEN, "\n================================================")
    print_color(Colors.GREEN, "Setup completed successfully!")
    print_color(Colors.GREEN, "================================================")
    print("\nThe following resources have been created:")
    print("  - Training data: data/raw/")
    print("  - Trained models: models/")
    print("  - Enhanced demo data: data/enhanced_demo/")
    print("  - Analytics reports: reports/enhanced_analysis/")
    print("  - User insights: reports/user_insights/")
    print("\nYou can now start developing with the Sleep Insights App.")
    
    # Start API server if requested
    if args.start_api:
        start_api_server()
    else:
        print("\nTo start the API server, run:")
        print("  uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")
    
    print("\nHappy coding!")

if __name__ == "__main__":
    main()