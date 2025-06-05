FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    swig \
    python3-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
# Try to install the potentially problematic packages using binary wheels only
RUN pip install --no-cache-dir --only-binary=:all: faiss-cpu==1.7.4 sentence-transformers==2.2.2 || echo "Could not install from binary wheels, will try from requirements.txt"
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create directories for data
RUN mkdir -p ./temp_onedrive_data ./stiahnute_zakony

# Make setup script executable
RUN chmod +x setup.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
