# Zákony Chat

A Streamlit application that allows users to chat with Slovak legal documents using LangChain and OpenAI models.

## Features

- Interactive chat interface for legal queries
- Dynamic loading of legal data from OneDrive
- Clean temporary data after use
- Optimized for Streamlit Cloud deployment

## Deployment

This application is ready to be deployed on Streamlit Cloud. Follow these steps:

1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Select this repository and the `app.py` file
4. Set the required secrets (like OpenAI API key) in the Streamlit dashboard

## Environment Variables

The following environment variables should be set in Streamlit deployment:

- `OPENAI_API_KEY`: Your OpenAI API key (can also be provided in the app UI)

## Requirements

All required packages are listed in `requirements.txt`.
