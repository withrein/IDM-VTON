# IDM-VTON GPU Deployment Guide

## Quick Start

1. **Upload deployment package to your rented GPU server**
   ```bash
   # Replace username and ip-address with your actual GPU server details
   scp idm_vton_deploy.zip username@ip-address:~/
   ```

2. **Connect to your GPU server**
   ```bash
   ssh username@ip-address
   ```

3. **Extract and run the deployment script**
   ```bash
   unzip idm_vton_deploy.zip
   chmod +x deploy_gpu.sh
   ./deploy_gpu.sh
   ```

4. **Access the application**
   - The script will display a URL like `https://xxx.gradio.app`
   - Open this URL on your mobile device to use the virtual try-on app

## Using the Application

1. Upload a photo of yourself (or use camera)
2. Upload a garment image
3. Describe the garment (e.g., "Red cotton t-shirt")
4. Click "Start Try-on"
5. Wait for processing (30-60 seconds)
6. View the result showing you wearing the garment

## Server Requirements

- CUDA-compatible GPU with at least 8GB VRAM
- Python 3.8+ with pip
- Internet connection for downloading models
- At least 10GB free disk space

See the full `DEPLOY_INSTRUCTIONS.md` and `MOBILE_USAGE.md` for detailed information.

## For vast.ai Users

If you're using vast.ai:

1. Create an instance with:
   - PyTorch template
   - CUDA 11.8+
   - At least 16GB VRAM

2. Connect using the provided SSH command in the vast.ai interface

3. Follow the deployment steps above 