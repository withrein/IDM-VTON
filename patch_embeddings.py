#!/usr/bin/env python3

import sys
import os
import importlib
import inspect

# Path to site-packages
site_packages = '/Users/rein/miniconda3/lib/python3.12/site-packages'
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

# Save a backup of our patch function that users can run directly
patch_script = f"""#!/usr/bin/env python3

# This script adds the PositionNet class to the diffusers embeddings module
import os
import sys

# Path to site-packages - update this to match your environment
site_packages = os.path.dirname(os.__file__) + '/../site-packages'
embeddings_path = os.path.join(site_packages, 'diffusers', 'models', 'embeddings.py')

# The PositionNet class to add
position_net_code = '''{position_net_code}'''

# Make sure this function exists
make_image_grid_function = '''{make_image_grid_function}'''

# Check if the file exists
if not os.path.exists(embeddings_path):
    print(f"Error: Could not find diffusers embeddings at {{embeddings_path}}")
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
    content += "\\n" + make_image_grid_function

# Add PositionNet class
content += "\\n" + position_net_code

# Write back the modified content
with open(embeddings_path, 'w') as f:
    f.write(content)

print(f"Successfully patched {{embeddings_path}} with PositionNet class!")
"""

with open('patch_diffusers.py', 'w') as f:
    f.write(patch_script)

print("Also created a standalone patch_diffusers.py script for future use.") 