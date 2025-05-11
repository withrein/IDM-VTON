#!/bin/bash

echo "==== IDM-VTON Remote Deployment Tool ===="
echo "This script helps you deploy the IDM-VTON project to a remote GPU server."

# Check if we have a server address
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <server_address> [user]"
  echo "Example: $0 user@123.45.67.89"
  exit 1
fi

SERVER_ADDRESS="$1"
USER_NAME="${2:-root}"  # Default to root if not specified

# Create a deployment folder
DEPLOY_DIR="idm_vton_deploy"
mkdir -p "$DEPLOY_DIR"

# Copy necessary files
echo "Preparing deployment package..."
cp setup_gpu.sh "$DEPLOY_DIR/"
cp -r patch_embeddings.py "$DEPLOY_DIR/" 2>/dev/null || :
cp -r gradio_demo "$DEPLOY_DIR/" 2>/dev/null || :

# Create a README
cat > "$DEPLOY_DIR/README.md" << 'EOL'
# IDM-VTON Remote Deployment

This package contains the necessary files to deploy the IDM-VTON virtual try-on system on a remote GPU server.

## Quick Start

Just run the setup script to get started:

```bash
chmod +x setup_gpu.sh
./setup_gpu.sh
```

This will:
1. Clone the repository
2. Install dependencies
3. Download model files
4. Set up the Gradio interface
5. Launch the mobile web app

When the app is running, you'll see a URL like `https://xxx.gradio.app` which you can access from your mobile device.
EOL

# Create a zip archive
echo "Creating deployment archive..."
ZIP_FILE="idm_vton_deploy.zip"
(cd "$DEPLOY_DIR" && zip -r "../$ZIP_FILE" *)

# Upload to server
echo "Deploying to $SERVER_ADDRESS..."
scp "$ZIP_FILE" "$USER_NAME@$SERVER_ADDRESS:~/"

# Execute remote setup
echo "Setting up remotely..."
ssh "$USER_NAME@$SERVER_ADDRESS" << EOF
  mkdir -p idm_vton
  cd idm_vton
  unzip ~/$ZIP_FILE
  chmod +x setup_gpu.sh
  echo "Setup files extracted. To start the application, run:"
  echo "cd ~/idm_vton && ./setup_gpu.sh"
EOF

echo "Deployment complete!"
echo "Connect to your server and run: cd ~/idm_vton && ./setup_gpu.sh"
echo "The app will then be available at a URL displayed in the terminal." 