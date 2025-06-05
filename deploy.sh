#!/bin/bash
# Deployment script for Zakony AI Chatbot

# Set default environment if not specified
DEPLOY_ENV=${1:-"docker"}

echo "Preparing to deploy Zakony AI to $DEPLOY_ENV environment..."

# Function to check for required tools
check_requirements() {
  for cmd in "$@"; do
    if ! command -v $cmd &> /dev/null; then
      echo "$cmd is required but not installed. Please install it first."
      exit 1
    fi
  done
}

# Deploy using Docker
deploy_docker() {
  echo "Deploying using Docker..."
  check_requirements docker docker-compose
  
  echo "Building and starting Docker containers..."
  docker-compose up --build -d
  
  echo "Docker deployment complete. Access the app at http://localhost:8501"
}

# Deploy to Railway
deploy_railway() {
  echo "Deploying to Railway..."
  check_requirements railway
  
  echo "Logging into Railway..."
  railway login
  
  echo "Deploying current directory to Railway..."
  railway up
  
  echo "Railway deployment initiated. Check your Railway dashboard for the deployment status and URL."
}

# Deploy to Render
deploy_render() {
  echo "Deploying to Render..."
  check_requirements curl
  
  echo "To deploy to Render, connect your GitHub repository to Render, or use:"
  echo "1. Login to the Render Dashboard"
  echo "2. Click 'New Web Service'"
  echo "3. Connect to your repository"
  echo "4. Select the repository with this project"
  echo "5. Use the following settings:"
  echo "   - Environment: Docker"
  echo "   - Build Command: [leave empty]"
  echo "   - Start Command: [leave empty]"
  
  echo "Would you like to open the Render Dashboard now? (y/n)"
  read answer
  if [[ $answer == "y" ]]; then
    open "https://dashboard.render.com/"
  fi
}

# Deploy to Fly.io
deploy_flyio() {
  echo "Deploying to Fly.io..."
  check_requirements flyctl
  
  echo "Logging into Fly.io..."
  flyctl auth login
  
  echo "Initializing Fly.io app..."
  flyctl launch --no-deploy
  
  echo "Deploying to Fly.io..."
  flyctl deploy
  
  echo "Fly.io deployment initiated. Check your Fly.io dashboard for the deployment status and URL."
}

# Create a simple bash script for local deployment
deploy_local() {
  echo "Setting up for local deployment..."
  
  # Create the local deployment script
  cat > deploy_local.sh << 'EOL'
#!/bin/bash
# Script for local deployment of Zakony AI

# Check Python version
if ! command -v python3 &> /dev/null; then
  echo "Python 3 is required but not installed. Please install it first."
  exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip wheel setuptools
pip install --only-binary=:all: faiss-cpu==1.7.4 sentence-transformers==2.2.2 || echo "Binary wheel installation failed, will try from requirements.txt"
pip install -r requirements.txt

# Run the application
echo "Starting the application..."
python run.py
EOL

  # Make the script executable
  chmod +x deploy_local.sh
  
  echo "Local deployment script created. Run ./deploy_local.sh to start the application."
}

# Display usage information
usage() {
  echo "Usage: $0 [environment]"
  echo
  echo "Available environments:"
  echo "  docker    - Deploy using Docker (default)"
  echo "  local     - Deploy locally with a virtual environment"
  echo "  railway   - Deploy to Railway"
  echo "  render    - Deploy to Render"
  echo "  flyio     - Deploy to Fly.io"
  echo
}

# Main execution
case "$DEPLOY_ENV" in
  "docker")
    deploy_docker
    ;;
  "local")
    deploy_local
    ;;
  "railway")
    deploy_railway
    ;;
  "render")
    deploy_render
    ;;
  "flyio")
    deploy_flyio
    ;;
  "help")
    usage
    exit 0
    ;;
  *)
    echo "Unknown environment: $DEPLOY_ENV"
    usage
    exit 1
    ;;
esac

echo "Deployment process completed."
