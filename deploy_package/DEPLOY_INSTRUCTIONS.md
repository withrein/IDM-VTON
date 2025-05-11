# IDM-VTON GPU Deployment Instructions

This document provides instructions for deploying the IDM-VTON virtual try-on system on a rented GPU server.

## Prerequisites

- A GPU server with CUDA support (ideally CUDA 11.8+)
- Python 3.8+ installed
- Git installed
- Internet connection to download models

## Option 1: Direct Deployment on GPU Server

1. **Copy the deployment script to your GPU server**
   - Transfer the `deploy_gpu.sh` script to your rented GPU
   - You can use SCP, rsync, or simply copy-paste the content

2. **Run the deployment script**
   ```bash
   chmod +x deploy_gpu.sh
   ./deploy_gpu.sh
   ```

3. **Access the application**
   - The script will print a shareable link (e.g., https://xxx.gradio.app)
   - Open this link in your browser or mobile device

## Option 2: Using the Remote Deployment Tool

If you want to deploy from your local machine to the GPU server:

1. **Update connection details**
   - Edit `setup_remote.sh` with your server information

2. **Run the remote setup script**
   ```bash
   ./setup_remote.sh username@your-server-ip
   ```

3. **Connect to the server and start the application**
   ```bash
   ssh username@your-server-ip
   cd ~/idm_vton
   ./setup_gpu.sh
   ```

## Troubleshooting

### Model Download Issues
If you encounter issues downloading the model files, you can manually download them from:
- DensePose model: https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl
- Human parsing models: 
  - https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx
  - https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx
- OpenPose model: https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth
- IP-Adapter: https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin
- Image encoder:
  - https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json
  - https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin
  - https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/preprocessor_config.json

### Memory Issues
If you encounter CUDA out of memory errors:
1. Reduce the batch size
2. Use fewer inference steps (adjust the slider in the UI)
3. Use a lower resolution for input images

### ImportError or ModuleNotFoundError
If you encounter import errors, you may need to install additional dependencies:
```bash
pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install timm==0.9.5 pytest==7.4.1
``` 