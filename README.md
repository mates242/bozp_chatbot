# BOZP Chatbot

A Streamlit-based chatbot application for querying legal documents related to occupational safety and health (BOZP).

## Overview

This application allows users to interact with a chatbot that can answer questions about safety regulations and laws. It uses LangChain and OpenAI models to process and retrieve information from legal documents.

## Features

- Document processing and embedding using OpenAI embeddings
- Vector search using FAISS
- Conversational interface with memory
- Support for preprocessing documents
- Integration with OneDrive for storing and retrieving preprocessed documents

## Requirements

- Python 3.8+
- OpenAI API key
- See `requirements.txt` for required packages

## Setup and Running

### Method 1: Direct Installation

1. Install system dependencies (on Ubuntu/Debian):
   ```bash
   sudo apt-get update && sudo apt-get install -y build-essential cmake pkg-config swig python3-dev
   ```
   On macOS:
   ```bash
   brew install cmake swig pkg-config
   ```

2. Install the Python requirements:
   ```bash
   pip install --only-binary=:all: faiss-cpu sentence-transformers
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Alternatively, use the run script which handles dependencies:
   ```bash
   python run.py
   ```

### Method 2: Using Docker (Recommended)

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. Or use Docker directly:
   ```bash
   docker build -t zakony-ai .
   docker run -p 8501:8501 zakony-ai
   ```

This method automatically installs all required system dependencies and Python packages.

## Documentation

The project consists of several key files:

- `app.py`: Main Streamlit application
- `onedrive_utils.py`: Utilities for downloading documents from OneDrive
- `preprocess_documents.py`: Functions for preprocessing legal documents
- `run.py`: Simplified script to run the application
- `Dockerfile` and `docker-compose.yml`: Container configuration for deployment
- `setup.sh`: Script to install system dependencies

## Deployment

### Using the Deployment Script

For simplified deployment, use the included `deploy.sh` script:

```bash
# Deploy using Docker (default)
./deploy.sh docker

# Deploy locally with a virtual environment
./deploy.sh local

# Deploy to Railway
./deploy.sh railway

# Deploy to Render
./deploy.sh render

# Deploy to Fly.io
./deploy.sh flyio

# Display help information
./deploy.sh help
```

### Cloud Deployment Options

1. **Streamlit Cloud**:
   - Connect your GitHub repository
   - Set the main file to `app.py`
   - Configure your OpenAI API key as a secret

2. **Railway/Render/Fly.io**:
   - Use the provided Dockerfile
   - Set the OpenAI API key as an environment variable
   - Deploy using the deployment script or platform-specific commands

3. **Heroku**:
   - The `.aptfile` and `Procfile` are already configured for Heroku
   - Add the buildpacks for apt and Python:
     ```bash
     heroku buildpacks:add --index 1 heroku-community/apt
     heroku buildpacks:add --index 2 heroku/python
     ```
   - Set the OpenAI API key as a config var

4. **AWS/Azure/GCP**:
   - Use the Docker container to deploy on container services (ECS, ACI, Cloud Run)
   - Alternatively, deploy on VMs with Docker installed

### Troubleshooting Deployment Issues

1. **Binary Dependencies**: 
   - If deployment fails with errors about `swig`, `cmake` or `pkg-config`, use the Docker deployment method which installs all system dependencies automatically.

2. **Package Installation Issues**:
   - If direct pip installation fails, use the `--only-binary=:all:` flag for problematic packages.
   - The Docker setup handles this automatically.

3. **Memory Issues**:
   - Ensure your deployment environment has at least 2GB of RAM as FAISS and language models require significant memory.
