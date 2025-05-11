#!/usr/bin/env python3

import os
import subprocess
import sys

def run_command(cmd):
    """Run a shell command and print output"""
    print(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Stream the output
    while True:
        output = process.stdout.readline().decode()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    return process.poll()

def setup_environment():
    """Setup the conda environment"""
    print("=== Setting up environment ===")
    
    # Create conda environment
    run_command("conda create -n idm python=3.10 -y")
    
    # Activate environment and install packages
    run_command("""
    conda activate idm &&
    conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y &&
    pip install accelerate==0.25.0 torchmetrics==1.2.1 tqdm==4.66.1 transformers==4.36.2 diffusers==0.25.0 &&
    pip install einops==0.7.0 bitsandbytes==0.39.0 scipy==1.11.1 opencv-python gradio==4.24.0 fvcore &&
    pip install cloudpickle omegaconf pycocotools basicsr av huggingface-hub==0.21.0
    """)
    
    print("Environment setup complete!")
    return True

def download_checkpoints():
    """Download required checkpoints"""
    print("=== Downloading required checkpoints ===")
    
    # Create checkpoint directories
    run_command("mkdir -p ckpt/densepose ckpt/humanparsing ckpt/image_encoder ckpt/ip_adapter ckpt/openpose/ckpts")
    
    # Download densepose model
    run_command("""
    mkdir -p ckpt/densepose &&
    wget -O ckpt/densepose/model_final_162be9.pkl https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl
    """)
    
    # Download human parsing models
    run_command("""
    mkdir -p ckpt/humanparsing &&
    wget -O ckpt/humanparsing/parsing_atr.onnx https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx &&
    wget -O ckpt/humanparsing/parsing_lip.onnx https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx
    """)
    
    # Download openpose model
    run_command("""
    mkdir -p ckpt/openpose/ckpts &&
    wget -O ckpt/openpose/ckpts/body_pose_model.pth https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth
    """)
    
    # Download IP-Adapter
    run_command("""
    mkdir -p ckpt/ip_adapter &&
    wget -O ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin
    """)
    
    # Download image encoder
    run_command("""
    mkdir -p ckpt/image_encoder &&
    wget -O ckpt/image_encoder/config.json https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json &&
    wget -O ckpt/image_encoder/pytorch_model.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin &&
    wget -O ckpt/image_encoder/preprocessor_config.json https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/preprocessor_config.json
    """)
    
    print("Checkpoint downloads complete!")
    return True

def patch_embeddings():
    """Patch diffusers embeddings if needed"""
    print("=== Patching diffusers embeddings if needed ===")
    
    # Run the patch script
    run_command("python patch_embeddings.py")
    
    print("Patching complete!")
    return True

def launch_demo():
    """Launch the mobile demo app"""
    print("=== Launching mobile demo ===")
    
    # Launch the mobile app with public access
    run_command("""
    conda activate idm &&
    cd /workspace/IDM-VTON &&
    python gradio_demo/mobile_app.py
    """)
    
    return True

def main():
    """Main execution function"""
    # Determine if we're on a system with conda properly setup 
    try:
        subprocess.run(['conda', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Conda not found or not properly configured. Please install Miniconda/Anaconda first.")
        return False
    
    # Clone repository if not already done
    if not os.path.exists("IDM-VTON"):
        print("=== Cloning repository ===")
        run_command("git clone https://github.com/yisol/IDM-VTON.git")
        os.chdir("IDM-VTON")
    elif not os.path.basename(os.getcwd()) == "IDM-VTON":
        os.chdir("IDM-VTON")
    
    # Run the setup steps
    setup_environment()
    download_checkpoints()
    patch_embeddings()
    launch_demo()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("Setup and launch completed successfully!")
    else:
        print("Setup failed!")
        sys.exit(1) 