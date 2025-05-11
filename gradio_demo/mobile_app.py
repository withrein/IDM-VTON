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

# Load example images
garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path, "human"))
human_list_path = [os.path.join(example_path, "human", human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict = {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

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
        
        # Example selector
        gr.Examples(
            examples=human_ex_list,
            inputs=imgs,
            label="Example Photos",
            examples_per_page=3
        )
        
        gr.Examples(
            examples=garm_list_path,
            inputs=garm_img,
            label="Example Garments",
            examples_per_page=3
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