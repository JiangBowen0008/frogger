import os
import numpy as np
import trimesh
import argparse

def convert_stl_to_obj(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all STL files in input directory
    stl_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.stl')]
    
    for stl_file in stl_files:
        input_path = os.path.join(input_dir, stl_file)
        output_file = os.path.splitext(stl_file)[0] + '.obj'
        output_path = os.path.join(output_dir, output_file)
        
        try:
            # Load the STL file using trimesh
            mesh_data = trimesh.load(input_path)
            
            # Export as OBJ
            mesh_data.export(output_path)
            print(f"Converted {stl_file} to {output_file}")
            
        except Exception as e:
            print(f"Error converting {stl_file}: {str(e)}")

if __name__ == "__main__":
    convert_stl_to_obj(
        "models/leap/meshes_stl",
        "models/leap/meshes_obj"
    )