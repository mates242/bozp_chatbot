import os
import subprocess
import sys
import importlib.util

def check_package(package_name):
    """Check if a package is installed."""
    return importlib.util.find_spec(package_name) is not None

def install_requirements():
    """Install requirements from requirements.txt file."""
    print("Installing required packages...")
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if os.path.exists(req_path):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
            print("All required packages installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install required packages.")
            return False
    else:
        print("requirements.txt not found.")
        return False

def main():
    """
    Launch the Streamlit app without requiring the user to run 'streamlit run' directly.
    """
    # Check if streamlit is installed, if not, install requirements
    if not check_package("streamlit"):
        print("Streamlit not found. Installing required packages...")
        if not install_requirements():
            print("Cannot proceed without required packages.")
            sys.exit(1)
    
    print("Starting Zákony AI Chatbot...")
    
    # Get the path to the app.py file
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    try:
        # Construct the command to run the Streamlit app
        cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--browser.serverAddress", "localhost"]
        
        # Execute the command
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down the application...")
    except Exception as e:
        print(f"Error starting the application: {e}")
        print("\nPlease ensure all dependencies are installed by running:")
        print("pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
