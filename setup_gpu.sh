#!/bin/bash

# Exit on error
set -e

echo "==== Setting Up IDM-VTON Virtual Try-on on GPU Server ===="

# Clone the repository if needed
if [ ! -d "IDM-VTON" ]; then
  echo "Cloning repository..."
  git clone https://github.com/yisol/IDM-VTON.git
  cd IDM-VTON
else
  cd IDM-VTON
  echo "Repository already exists, using existing copy."
fi

# Install dependencies
echo "Installing required packages..."
pip install diffusers==0.25.0 huggingface-hub==0.21.0 transformers==4.36.2 gradio==4.24.0 accelerate==0.25.0
pip install torch torchvision torchaudio numpy opencv-python einops bitsandbytes==0.39.0 scipy fvcore cloudpickle omegaconf pycocotools onnxruntime

# Create directories for checkpoints
echo "Creating directories for checkpoints..."
mkdir -p ckpt/densepose
mkdir -p ckpt/humanparsing
mkdir -p ckpt/openpose/ckpts
mkdir -p ckpt/ip_adapter
mkdir -p ckpt/image_encoder

# Download necessary model files
echo "Downloading model files (this may take a while)..."
# DensePose model
wget -O ckpt/densepose/model_final_162be9.pkl https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl

# Human parsing models
wget -O ckpt/humanparsing/parsing_atr.onnx https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx
wget -O ckpt/humanparsing/parsing_lip.onnx https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx

# OpenPose model
wget -O ckpt/openpose/ckpts/body_pose_model.pth https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth

# IP-Adapter
wget -O ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin

# Image encoder
wget -O ckpt/image_encoder/config.json https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json
wget -O ckpt/image_encoder/pytorch_model.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin
wget -O ckpt/image_encoder/preprocessor_config.json https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/preprocessor_config.json

# Create the patch_embeddings.py file if not already exists
if [ ! -f "patch_embeddings.py" ]; then
  echo "Creating patch_embeddings.py file..."
  cat > patch_embeddings.py << 'EOL'
#!/usr/bin/env python3

import sys
import os
import importlib
import inspect

# Path to site-packages
site_packages = None
for path in sys.path:
    if 'site-packages' in path:
        site_packages = path
        break

if not site_packages:
    print("Could not find site-packages directory")
    sys.exit(1)

embeddings_path = os.path.join(site_packages, 'diffusers', 'models', 'embeddings.py')

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

# Check if the file exists
if not os.path.exists(embeddings_path):
    print(f"Error: Could not find diffusers embeddings at {embeddings_path}")
    sys.exit(1)

# Read the existing content
with open(embeddings_path, 'r') as f:
    content = f.read()

# Check if PositionNet already exists
if 'class PositionNet' in content:
    print("PositionNet class already exists. No need to patch.")
    sys.exit(0)

# Check if make_image_grid exists
if 'def make_image_grid' not in content:
    # Add make_image_grid function
    content += "\n" + make_image_grid_function

# Add PositionNet class
content += "\n" + position_net_code

# Write back the modified content
with open(embeddings_path, 'w') as f:
    f.write(content)

print(f"Successfully patched {embeddings_path} with PositionNet class!")
EOL
fi

# Run the patch script
echo "Running patch script to add PositionNet class if needed..."
python patch_embeddings.py

# Ensure the mobile_app.py exists
echo "Checking for mobile_app.py..."
if [ ! -d "gradio_demo" ]; then
  mkdir -p gradio_demo
fi

# Create mobile_app.py if needed
if [ ! -f "gradio_demo/mobile_app.py" ]; then
  echo "Creating mobile_app.py..."
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

# Set paths based on deployment environment
base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

# Load models
def load_models():
    # Load UNet model
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)
    
    # Load tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )
    
    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
    
    # Load text encoders
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    
    # Load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        base_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    
    # Load UNet Encoder
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    )
    
    # Load parsing and pose models
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    
    # Set models to evaluation mode
    UNet_Encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    # Define tensor transform
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Initialize pipeline
    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe.unet_encoder = UNet_Encoder
    
    return pipe, parsing_model, openpose_model, tensor_transform

# Initialize model variables at module level
pipe, parsing_model, openpose_model, tensor_transform = load_models()

def start_tryon(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed, progress=gr.Progress()):
    progress(0, desc="Loading models...")
    
    # Move models to device
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)
    
    # Process input images
    progress(0.1, desc="Processing images...")
    garm_img = garm_img.convert("RGB").resize((768,1024))
    human_img_orig = dict["background"].convert("RGB")
    
    # Crop image if requested
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))
    
    progress(0.2, desc="Generating mask...")
    # Generate or use provided mask
    if is_checked:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
    
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)
    
    progress(0.3, desc="Processing pose...")
    # Process pose information
    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    # Create arguments for pose detection
    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', device))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:,:,::-1]
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    progress(0.4, desc="Generating try-on image...")
    # Generate try-on image
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
            with torch.inference_mode():
                # Encode prompt
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                
                # Encode garment prompt
                prompt = "a photo of " + garment_des
                if not isinstance(prompt, List):
                    prompt = [prompt] * 1
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * 1
                
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )
                
                # Process input images for model
                pose_img = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
                garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
                
                # Set generator seed if provided
                generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                
                # Run inference
                progress(0.5, desc=f"Running inference ({denoise_steps} steps)...")
                images = pipe(
                    prompt_embeds=prompt_embeds.to(device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img.to(device, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                    cloth=garm_tensor.to(device, torch.float16),
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768,1024)),
                    guidance_scale=2.0,
                )[0]
    
    progress(0.9, desc="Finalizing...")
    # Process and return output image
    if is_checked_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray

# Create example directories if they don't exist
if not os.path.exists(example_path):
    os.makedirs(os.path.join(example_path, "cloth"), exist_ok=True)
    os.makedirs(os.path.join(example_path, "human"), exist_ok=True)

# Create Gradio interface - mobile optimized
def create_interface():
    with gr.Blocks(title="IDM-VTON Mobile", css="footer {display:none !important}") as demo:
        gr.Markdown("# IDM-VTON 👕👔👚")
        gr.Markdown("Virtual Try-on for mobile devices")
        
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
        
        # Try-on button
        try_button = gr.Button("Start Try-on", variant="primary", scale=1)
        try_button.click(
            fn=start_tryon,
            inputs=[imgs, garm_img, prompt, is_checked, is_checked_crop, denoise_steps, seed],
            outputs=[image_out, masked_img],
            api_name='tryon'
        )
    
    return demo

# Create and launch the interface
mobile_interface = create_interface()

# Main function to run the app
if __name__ == "__main__":
    # Launch with server configuration optimized for mobile access
    mobile_interface.launch(
        server_name="0.0.0.0",  # Bind to all interfaces
        server_port=7860,       # Default Gradio port
        share=True,             # Generate shareable link
        enable_queue=True,      # Enable queuing for better performance
        max_threads=40,         # Limit threads for mobile performance
        show_error=True,        # Show errors for debugging
        inbrowser=False         # Don't open browser on server
    )
EOL
fi

# Create example directories
mkdir -p gradio_demo/example/cloth
mkdir -p gradio_demo/example/human

# Launch the demo
echo "Starting the mobile demo app..."
echo "This will download the necessary models on first run (may take some time)..."
python gradio_demo/mobile_app.py 