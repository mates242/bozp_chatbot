#!/bin/bash
# This script installs necessary system dependencies for the application

echo "Installing required system dependencies..."

# Update package lists
apt-get update -y || true

# Install build essentials, cmake, swig and pkg-config
apt-get install -y build-essential cmake swig pkg-config || true

# Install Python development headers
apt-get install -y python3-dev || true

# Create a .aptfile to specify system dependencies if needed by the hosting platform
cat > .aptfile << EOF
build-essential
cmake
swig
pkg-config
python3-dev
EOF

echo "System dependencies installation completed"

# Now install Python packages
pip install --upgrade pip
pip install --only-binary=:all: -r requirements.txt

echo "Python dependencies installation completed"
