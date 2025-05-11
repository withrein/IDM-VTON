#!/usr/bin/env python3

import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"Error importing torch: {e}")

try:
    import diffusers
    print(f"Diffusers version: {diffusers.__version__}")
    
    # Check for PositionNet in diffusers.models.embeddings
    try:
        from diffusers.models.embeddings import PositionNet
        print("PositionNet was found in diffusers.models.embeddings")
    except ImportError:
        print("PositionNet is NOT found in diffusers.models.embeddings")
        
        # Print all classes in embeddings module
        import inspect
        from diffusers.models import embeddings
        
        print("Available classes in diffusers.models.embeddings:")
        for name, obj in inspect.getmembers(embeddings):
            if inspect.isclass(obj):
                print(f"  - {name}")
except ImportError as e:
    print(f"Error importing diffusers: {e}")

# Look for the error in the src module
try:
    import src.unet_hacked_garmnet
    print("Successfully imported src.unet_hacked_garmnet")
except ImportError as e:
    print(f"Error importing src.unet_hacked_garmnet: {e}")
    
    # Check the actual import line
    try:
        from diffusers.models.embeddings import TimestepEmbedding, Timesteps, GaussianFourierProjection
        print("Successfully imported TimestepEmbedding, Timesteps, GaussianFourierProjection")
    except ImportError as e:
        print(f"Error: {e}") 