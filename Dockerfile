FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV for package management
RUN curl -sSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"
    
# Copy requirements and lock files
COPY requirements.txt uv.lock ./

# Install dependencies using uv sync
RUN uv sync
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV MODELS_DIR=/app/models

# Default command to run Sleep Advisor
CMD ["python", "scripts/sleep_advisor.py", "--form-data-file", "data/raw/sleep_form_data.csv", "--historical-data", "data/processed/historical_sleep_data.csv", "--output-dir", "data/recommendations"]