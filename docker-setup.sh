#!/bin/bash
# docker-setup.sh - Sleep Insights App Docker Setup Script
# This script prepares the Sleep Insights app using Docker containers

set -e  # Exit on error

# ANSI color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Sleep Insights App - Docker Setup Script${NC}"
echo -e "${BLUE}================================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker and try again.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose and try again.${NC}"
    exit 1
fi

# Create required directories
echo -e "\n${YELLOW}Creating required directories...${NC}"
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/enhanced_demo
mkdir -p data/logs
mkdir -p models
mkdir -p reports/enhanced_analysis
mkdir -p reports/user_insights
mkdir -p logs

# Create logs directory for container output
mkdir -p container_logs

# Create a Dockerfile for the setup if it doesn't exist
if [ ! -f "Dockerfile.setup" ]; then
    echo -e "\n${YELLOW}Creating Dockerfile.setup...${NC}"
    cat > Dockerfile.setup << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch

# Copy all files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Set entry point for flexibility
ENTRYPOINT ["bash"]
EOF
fi

# Create a docker-compose file for setup if it doesn't exist
if [ ! -f "docker-compose.setup.yml" ]; then
    echo -e "\n${YELLOW}Creating docker-compose.setup.yml...${NC}"
    cat > docker-compose.setup.yml << 'EOF'
version: '3.8'

services:
  setup:
    build:
      context: .
      dockerfile: Dockerfile.setup
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
      - ./logs:/app/logs
      - ./container_logs:/app/container_logs
    environment:
      - PYTHONPATH=/app
      - PYTHONUNBUFFERED=1
    command: -c "echo 'Running in container...'"

  data_generation:
    extends: setup
    command: -c "python data_generation_script.py 2>&1 | tee /app/container_logs/data_generation.log"

  train_models:
    extends: setup
    command: -c "python train_models.py --data-dir data/raw --output-dir models 2>&1 | tee /app/container_logs/train_models.log"

  enhanced_demo:
    extends: setup
    command: -c "python run_enhanced_demo.py --output-dir data/enhanced_demo --user-count 200 --wearable-percentage 30 2>&1 | tee /app/container_logs/enhanced_demo.log"

  sleep_advisor:
    extends: setup
    command: -c "python sleep_advisor.py --form-data-file data/enhanced_demo/data/sleep_data.csv --historical-data data/enhanced_demo/data/sleep_data.csv --output-dir data/enhanced_demo/recommendations 2>&1 | tee /app/container_logs/sleep_advisor.log"

  enhanced_analytics:
    extends: setup
    command: -c "python analyze_enhanced_data.py --data-dir data/enhanced_demo/data --output-dir reports/enhanced_analysis 2>&1 | tee /app/container_logs/enhanced_analytics.log"

  api:
    extends: setup
    ports:
      - "8000:8000"
    command: -c "python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
EOF
fi

# Build the setup container
echo -e "\n${YELLOW}Building Docker containers...${NC}"
docker-compose -f docker-compose.setup.yml build

# Run step 1: Generate training data
echo -e "\n${YELLOW}Step 1: Generating training data...${NC}"
docker-compose -f docker-compose.setup.yml run --rm data_generation
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data generation completed successfully${NC}"
else
    echo -e "${RED}✗ Data generation failed. Check container_logs/data_generation.log for details${NC}"
    exit 1
fi

# Run step 2: Train models
echo -e "\n${YELLOW}Step 2: Training models...${NC}"
docker-compose -f docker-compose.setup.yml run --rm train_models
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Model training completed successfully${NC}"
else
    echo -e "${RED}✗ Model training failed. Check container_logs/train_models.log for details${NC}"
    echo -e "${YELLOW}Continuing with fallback methods...${NC}"
fi

# Run step 3: Run enhanced demo
echo -e "\n${YELLOW}Step 3: Running enhanced demo...${NC}"
docker-compose -f docker-compose.setup.yml run --rm enhanced_demo
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Enhanced demo completed successfully${NC}"
else
    echo -e "${RED}✗ Enhanced demo failed. Check container_logs/enhanced_demo.log for details${NC}"
    exit 1
fi

# Run step 4: Analyze enhanced demo
echo -e "\n${YELLOW}Step 4: Analyzing enhanced demo data...${NC}"
docker-compose -f docker-compose.setup.yml run --rm sleep_advisor
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Sleep advisor analysis completed successfully${NC}"
else
    echo -e "${RED}✗ Sleep advisor analysis failed. Check container_logs/sleep_advisor.log for details${NC}"
    echo -e "${YELLOW}Continuing with other analyses...${NC}"
fi

# Run step 5: Run enhanced analytics
echo -e "\n${YELLOW}Step 5: Running enhanced analytics...${NC}"
docker-compose -f docker-compose.setup.yml run --rm enhanced_analytics
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Enhanced analytics completed successfully${NC}"
else
    echo -e "${RED}✗ Enhanced analytics failed. Check container_logs/enhanced_analytics.log for details${NC}"
    echo -e "${YELLOW}Some analyses may be incomplete...${NC}"
fi

# Run step 6: Generate user insights for a sample user
echo -e "\n${YELLOW}Step 6: Generating sample user insights...${NC}"

# Find a sample user ID from the generated data
SAMPLE_USER_ID=$(docker-compose -f docker-compose.setup.yml run --rm setup bash -c "head -n 10 data/enhanced_demo/data/users.csv | tail -n 1 | cut -d',' -f1" | tr -d '\r')

if [ -n "$SAMPLE_USER_ID" ]; then
    echo -e "Selected user ID: $SAMPLE_USER_ID"
    docker-compose -f docker-compose.setup.yml run --rm setup bash -c "python user-insights-generator.py --user-id $SAMPLE_USER_ID --data-dir data/enhanced_demo/data --output-dir reports/user_insights 2>&1 | tee /app/container_logs/user_insights.log"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ User insights generated successfully${NC}"
    else
        echo -e "${RED}✗ User insights generation failed. Check container_logs/user_insights.log for details${NC}"
    fi
else
    echo -e "${RED}No sample user found. Skipping user insights generation.${NC}"
fi

# Ask to start API server
echo -e "\n${YELLOW}Would you like to start the API server now? (y/n)${NC}"
read -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Starting API server...${NC}"
    echo -e "API will be available at http://localhost:8000"
    echo -e "To stop the server, press Ctrl+C"
    docker-compose -f docker-compose.setup.yml up api
else
    echo -e "\n${YELLOW}API server not started. You can start it later with:${NC}"
    echo -e "  docker-compose -f docker-compose.setup.yml up api"
fi

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Docker setup completed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "\nThe following resources have been created:"
echo -e "  - Training data: data/raw/"
echo -e "  - Trained models: models/"
echo -e "  - Enhanced demo data: data/enhanced_demo/"
echo -e "  - Analytics reports: reports/enhanced_analysis/"
echo -e "  - User insights: reports/user_insights/"
echo -e "\nYou can now start developing with the Sleep Insights App in Docker containers."
echo -e "Happy coding!\n"