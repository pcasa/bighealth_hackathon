FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies with standard pip
RUN pip install -r requirements.txt

# Install torch separately (since it needs special handling)
RUN pip install torch

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python", "scripts/sleep_advisor.py"]