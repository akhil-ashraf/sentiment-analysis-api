# Use Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run the API with proper host binding
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]