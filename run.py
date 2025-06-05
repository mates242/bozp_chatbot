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
            # Try to install using binary wheels only first (faster and more reliable)
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--only-binary=:all:", 
                "--upgrade", "pip", "wheel", "setuptools"
            ])
            print("Upgraded pip, wheel, and setuptools")
            
            try:
                # Try to install faiss-cpu and sentence-transformers using binary wheels only
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--only-binary=:all:", 
                    "faiss-cpu==1.7.4", "sentence-transformers==2.2.2"
                ])
                print("Successfully installed faiss-cpu and sentence-transformers using binary wheels")
            except subprocess.CalledProcessError:
                print("Warning: Could not install faiss-cpu or sentence-transformers from binary wheels")
                print("The application will attempt to run without these packages")
            
            # Install the rest of the requirements
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
            print("All required packages installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install required packages: {e}")
            return False
    else:
        print("requirements.txt not found.")
        return False

def check_system_dependencies():
    """Check if we can access system tools needed for building certain packages."""
    try:
        # Check for Mac-specific tools
        if sys.platform == 'darwin':
            # macOS: Check for Homebrew as a prerequisite
            brew_exists = subprocess.run(["which", "brew"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
            if brew_exists:
                print("Homebrew detected on macOS")
                return True
            else:
                print("Homebrew not detected. Some dependencies may not build properly.")
                return False
        # For other platforms, assume it's okay 
        return True
    except Exception as e:
        print(f"Error checking system dependencies: {e}")
        return False

def main():
    """
    Launch the Streamlit app without requiring the user to run 'streamlit run' directly.
    """
    # Check system dependencies first
    check_system_dependencies()
    
    # Check if setup.sh exists and run it if possible
    setup_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup.sh")
    if os.path.exists(setup_script) and sys.platform != 'win32':
        try:
            print("Running setup script to install system dependencies...")
            # Make setup.sh executable
            os.chmod(setup_script, 0o755)
            # Try to run setup.sh
            subprocess.run(['bash', setup_script], check=False)
        except Exception as e:
            print(f"Could not run setup script: {e}")
    
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
