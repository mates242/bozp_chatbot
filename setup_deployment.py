#!/usr/bin/env python3
"""
Setup script to prepare the app for Streamlit deployment
"""
import os
import sys
import subprocess

def check_requirements():
    print("Checking requirements...")
    try:
        import streamlit
        import langchain
        import openai
        import faiss
        print("Core requirements already installed.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed.")

def setup_streamlit_config():
    print("Setting up Streamlit configuration...")
    os.makedirs(".streamlit", exist_ok=True)
    
    config_path = os.path.join(".streamlit", "config.toml")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write("""[theme]
primaryColor="#FF4B4B"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

[server]
maxUploadSize=50
enableXsrfProtection=true
enableCORS=false

[browser]
gatherUsageStats=false
""")
        print("Streamlit configuration created.")
    else:
        print("Streamlit configuration already exists.")

def check_onedrive_access():
    print("Testing OneDrive access...")
    try:
        from onedrive_utils import clean_onedrive_url
        test_url = "https://1drv.ms/f/c/6c73abce92e1e313/ElphONm-mWFEsEYR1CqYGBIBzVSueQFLDmdWGl4-3uEaLQ?e=nJb4kD"
        cleaned_url = clean_onedrive_url(test_url)
        print(f"OneDrive access working. Test URL resolved: {cleaned_url}")
    except Exception as e:
        print(f"Error testing OneDrive access: {e}")
        print("Please check your OneDrive configuration.")

if __name__ == "__main__":
    print("Setting up Zákony Chat for Streamlit deployment...")
    check_requirements()
    setup_streamlit_config()
    check_onedrive_access()
    print("\nSetup complete! Your app is ready for Streamlit deployment.")
    print("To run locally: streamlit run app.py")
    print("To deploy on Streamlit Cloud, follow the instructions in README.md")
