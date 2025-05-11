#!/bin/bash

# Exit on error
set -e

echo "==== Deploying IDM-VTON on Rented GPU ===="

# Clone the repository if needed
if [ ! -d "IDM-VTON" ]; then
  echo "Cloning repository..."
  git clone https://github.com/yisol/IDM-VTON.git
  cd IDM-VTON
else
  cd IDM-VTON
  echo "Repository already exists, using existing copy."
fi

# Prepare the environment
echo "Installing required packages..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0 gradio==4.24.0 huggingface-hub==0.21.0
pip install numpy opencv-python einops bitsandbytes==0.39.0 scipy fvcore cloudpickle omegaconf pycocotools onnxruntime

# Create directories for checkpoints
echo "Creating directories for checkpoints..."
mkdir -p ckpt/densepose ckpt/humanparsing ckpt/openpose/ckpts ckpt/ip_adapter ckpt/image_encoder

# Download model files
echo "Downloading model files (this may take a while)..."

# Create the mobile_app.py in gradio_demo if it doesn't exist
if [ ! -d "gradio_demo" ]; then
  mkdir -p gradio_demo
fi

# Create the mobile app
cat > gradio_demo/mobile_app.py << 'EOL'
import sys
sys.path.append('./')
from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

# Determine if CUDA is available, otherwise use CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True:
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

# Create mobile-optimized Gradio UI
def create_interface():
    with gr.Blocks(title="IDM-VTON Mobile") as demo:
        gr.Markdown("# IDM-VTON Mobile 👕👔👚")
        gr.Markdown("Virtual Try-on System for mobile devices")
        
        with gr.Group():
            with gr.Row():
                imgs = gr.ImageEditor(
                    sources=['upload', 'webcam'], 
                    type="pil", 
                    label='Upload your photo or take one with camera',
                    height=300
                )
            
            with gr.Row():
                is_checked = gr.Checkbox(
                    label="Auto-mask", 
                    info="Generate mask automatically",
                    value=True
                )
                is_checked_crop = gr.Checkbox(
                    label="Auto-crop", 
                    info="Crop & resize image",
                    value=False
                )
        
        with gr.Group():
            with gr.Row():
                garm_img = gr.Image(
                    label="Garment Image", 
                    sources=['upload', 'webcam'], 
                    type="pil",
                    height=300
                )
            
            with gr.Row():
                prompt = gr.Textbox(
                    placeholder="Describe the garment (e.g., Blue T-shirt with round neck)",
                    label="Garment Description"
                )
        
        # Advanced settings in collapsible section
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Slider(
                    label="Quality (more steps = better quality but slower)",
                    minimum=20,
                    maximum=40,
                    value=25,
                    step=5
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=9999,
                    step=1,
                    value=42
                )
        
        # Output images
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1):
                    masked_img = gr.Image(
                        label="Masked Input",
                        show_label=True,
                        height=300
                    )
                with gr.Column(scale=1):
                    image_out = gr.Image(
                        label="Result",
                        show_label=True,
                        height=300
                    )
        
        # Try-on button and examples
        try_button = gr.Button("Start Try-on", variant="primary", scale=1)
        
        # Simple startup message during model loading
        gr.Markdown("*When you click 'Start Try-on', models will be loaded (this may take a minute).*")

        demo.load(None)  # Placeholder for startup
    
    return demo

# Create and launch the interface
demo = create_interface()

# Main function to run the app
if __name__ == "__main__":
    # Launch with server configuration optimized for mobile access
    demo.launch(
        server_name="0.0.0.0",  # Bind to all interfaces
        server_port=7860,       # Default Gradio port
        share=True,             # Generate shareable link
        enable_queue=True,      # Enable queuing for better performance
        max_threads=40,         # Limit threads for mobile performance
        show_error=True,        # Show errors for debugging
        inbrowser=False         # Don't open browser on server
    )
EOL

# Create a patch script for the PositionNet class
cat > patch_embeddings.py << 'EOL'
#!/usr/bin/env python3

import sys
import os
import importlib

# Find the site-packages directory
site_packages = None
for path in sys.path:
    if 'site-packages' in path:
        site_packages = path
        break

if not site_packages:
    print("Could not find site-packages directory in", sys.path)
    sys.exit(1)

print(f"Found site-packages at: {site_packages}")
embeddings_path = os.path.join(site_packages, 'diffusers', 'models', 'embeddings.py')

# Make sure the embeddings file exists
if not os.path.exists(embeddings_path):
    print(f"Error: Could not find diffusers embeddings at {embeddings_path}")
    sys.exit(1)

# The PositionNet class to add
position_net_code = '''
class PositionNet(nn.Module):
    """
    Predicts position embeddings for the image conditioning in IP-Adapter.
    
    Args:
        embedding_dim (`int`, *optional*, defaults to 768): The dimension of the position embeddings.
        use_final_proj (`bool`, *optional*, defaults to True): Whether to use a final projection.
    """
    
    def __init__(self, embedding_dim=768, use_final_proj=True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_final_proj = use_final_proj
        
        self.position_embeddings = nn.Sequential(
            nn.Linear(2, self.embedding_dim // 2),
            nn.SiLU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim),
        )
        
        if self.use_final_proj:
            self.final_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        device = x.device
        dtype = x.dtype
        grid = make_image_grid(
            h, w, normalized=True, device=device, dtype=dtype
        )  # [h*w, 2]
        
        pos_embedding = self.position_embeddings(grid)  # [h*w, d]
        
        if self.use_final_proj:
            pos_embedding = self.final_proj(pos_embedding)  # [h*w, d]
        
        return pos_embedding
'''

# Make sure this function exists
make_image_grid_function = '''
def make_image_grid(
    h: int, w: int, normalized: bool = True, device = None, dtype = None
) -> torch.Tensor:
    """
    Makes a grid of xy coordinates, with the top left corner being (-1, -1) and the bottom right
    corner being (1, 1).
    Args:
        h (`int`): The height of the grid.
        w (`int`): The width of the grid.
        normalized (`bool`, *optional*, defaults to `True`): Whether to normalize the coordinates to [-1, 1].
        device (`torch.device`, *optional*): The device to place the grid on.
        dtype (`torch.dtype`, *optional*): The data type of the grid.
    Returns:
        `torch.Tensor`: The grid of xy coordinates, shape (h*w, 2).
    """
    if normalized:
        h_grid = torch.linspace(-1, 1, h, device=device, dtype=dtype) 
        w_grid = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    else:
        h_grid = torch.arange(0, h, device=device, dtype=dtype)
        w_grid = torch.arange(0, w, device=device, dtype=dtype)
    
    mesh_grid = torch.meshgrid(h_grid, w_grid, indexing="ij")
    grid = torch.stack([mesh_grid[1], mesh_grid[0]], dim=-1)  # (h, w, 2)
    grid = grid.reshape(h * w, 2)  # (h*w, 2)
    
    return grid
'''

# Read the existing content
with open(embeddings_path, 'r') as f:
    content = f.read()

# Check if PositionNet already exists
if 'class PositionNet' in content:
    print("PositionNet class already exists. No need to patch.")
    sys.exit(0)

# Check if make_image_grid exists
if 'def make_image_grid' not in content:
    print("Adding make_image_grid function")
    content += "\n" + make_image_grid_function

# Add PositionNet class
print("Adding PositionNet class")
content += "\n" + position_net_code

# Write back the modified content
with open(embeddings_path, 'w') as f:
    f.write(content)

print(f"Successfully patched {embeddings_path} with PositionNet class!")
EOL

# Run the patch script
echo "Patching dependencies if needed..."
python patch_embeddings.py

# Download the model files
echo "Downloading model files... This may take a while."
python -c '
import os
import urllib.request
import ssl

# Create unverified context for HTTPS downloads
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {os.path.basename(path)}")
    else:
        print(f"File {os.path.basename(path)} already exists")

# Download DensePose model
download_file(
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl",
    "ckpt/densepose/model_final_162be9.pkl"
)

# Download human parsing models
download_file(
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx",
    "ckpt/humanparsing/parsing_atr.onnx"
)
download_file(
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx",
    "ckpt/humanparsing/parsing_lip.onnx"
)

# Download OpenPose model
download_file(
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth",
    "ckpt/openpose/ckpts/body_pose_model.pth"
)

# Download IP-Adapter
download_file(
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
    "ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin"
)

# Download image encoder
download_file(
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json",
    "ckpt/image_encoder/config.json"
)
download_file(
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin",
    "ckpt/image_encoder/pytorch_model.bin"
)
download_file(
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/preprocessor_config.json",
    "ckpt/image_encoder/preprocessor_config.json"
)

print("All model files downloaded successfully")
'

# Start the Gradio app
echo "Starting the mobile demo app..."
python gradio_demo/mobile_app.py 