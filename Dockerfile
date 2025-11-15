FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# - git: required for uv-dynamic-versioning in Agentics package
# - build-essential, g++: required for building hnswlib and other C++ extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        g++ \
        gcc \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Copy application files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.baseUrlPath=/agentics-finance"]