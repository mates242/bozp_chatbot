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

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```

3. Alternatively, use the run script:
   ```bash
   python run.py
   ```

## Documentation

The project consists of several key files:

- `app.py`: Main Streamlit application
- `onedrive_utils.py`: Utilities for downloading documents from OneDrive
- `preprocess_documents.py`: Functions for preprocessing legal documents
- `run.py`: Simplified script to run the application
