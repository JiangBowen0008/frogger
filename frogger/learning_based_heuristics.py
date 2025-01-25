import os, sys
import torch
import numpy as np
import open3d
import trimesh
import pickle

from contactdb.models.voxnet import DiverseVoxNet as VoxNet
from contactgen.model import ContactGenModel
from contactgen.utils.cfg_parser import Config

def remap_storage(storage, location):
    return storage

class ContactDBHeuristic:
    def __init__(self, 
                 checkpoint_path="/home/bowenj/Projects/DexFun/third_parties/contactdb_prediction/data/checkpoints/use_voxnet_diversenet_release/checkpoint_model_86_val_loss=0.01107167.pth",
                 grid_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        original_module = sys.modules.get('models', None)
        sys.modules['models'] = sys.modules.get('contactdb.models')
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            from contactdb.models.voxnet import DiverseVoxNet as VoxNet
            self.model = VoxNet(n_ensemble=checkpoint.n_ensemble, droprate=0.1)
            self.model.voxnet.load_state_dict(checkpoint.voxnet.state_dict())
        finally:
            if original_module is None:
                del sys.modules['models']
            else:
                sys.modules['models'] = original_module
                
        self.model.to(self.device)
        self.model.eval()
        self.grid_size = grid_size
        self.current_object = None
        self.vertices = None
        self.normals = None

    def load_object(self, mesh):
        self.vertices = np.array(mesh.vertices)
        self.normals = np.array(mesh.vertex_normals)
        
        min_bound = self.vertices.min(axis=0) 
        max_bound = self.vertices.max(axis=0)
        voxel_size = (max_bound - min_bound).max() / (self.grid_size - 1)
        
        # Get voxel coordinates
        voxel_coords = ((self.vertices - min_bound) / voxel_size).astype(int)
        x, y, z = voxel_coords.T
        pts = self.vertices.T
        
        # Center and scale
        offset = (pts.max(1, keepdims=True) + pts.min(1, keepdims=True)) / 2
        pts -= offset
        scale = max(pts.max(1) - pts.min(1)) / 2
        pts /= scale
        pts = np.vstack((np.ones(pts.shape[1]), pts, scale*np.ones(pts.shape[1])))
        
        # Center in grid
        offset_x = (self.grid_size - x.max() - 1) // 2
        offset_y = (self.grid_size - y.max() - 1) // 2
        offset_z = (self.grid_size - z.max() - 1) // 2
        x += offset_x
        y += offset_y
        z += offset_z
        
        # Ensure valid coordinates
        mask = (x >= 0) & (x < self.grid_size) & (y >= 0) & (y < self.grid_size) & (z >= 0) & (z < self.grid_size)
        x, y, z = x[mask], y[mask], z[mask]
        pts = pts[:, mask]
        self.valid_normals = self.normals[mask]

        # Create and store occupancy grid
        geom = np.zeros((5, self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        geom[:, z, y, x] = pts
        self.current_object = {
            'geom': torch.FloatTensor(geom).unsqueeze(0).to(self.device),
            'coords': (x, y, z)
        }

    def query(self, visualize=False):
        if self.current_object is None:
            raise ValueError("No object loaded. Call load_object first.")
            
        with torch.no_grad():
            predictions = self.model(self.current_object['geom'])
        
        predictions = predictions.cpu().numpy().squeeze()
        x, y, z = self.current_object['coords']
        binary_pred = np.argmax(predictions[0], axis=0)
        heatmap = binary_pred[z, y, x].astype(float)
        
        if visualize:
            pc = open3d.geometry.PointCloud()
            pc.points = open3d.utility.Vector3dVector(np.vstack((x, y, z)).T)
            colors = np.zeros((len(x), 3))
            colors[:, 0] = heatmap
            colors[:, 2] = 1 - heatmap
            pc.colors = open3d.utility.Vector3dVector(colors)
            open3d.visualization.draw_geometries([pc])
            
        return heatmap, self.valid_normals

class ContactGenHeuristic:
    def __init__(self, 
                 checkpoint_path="/home/bowenj/Projects/DexFun/third_parties/contactgen/checkpoint/checkpoint.pt", 
                 cfg_path="/home/bowenj/Projects/DexFun/third_parties/contactgen/contactgen/configs/default.yaml"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = Config(default_cfg_path=cfg_path)
        self.model = ContactGenModel(cfg).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval()
        self.current_object = None

    def load_object(self, mesh, n_points=2048):
        samples = trimesh.sample.sample_surface(mesh, n_points)
        verts = samples[0].astype(np.float32)
        normals = mesh.face_normals[samples[1]].astype(np.float32)
        
        self.current_object = {
            'verts': torch.from_numpy(verts).unsqueeze(0).to(self.device),
            'normals': torch.from_numpy(normals).unsqueeze(0).to(self.device)
        }

    def query(self, visualize=False):
        if self.current_object is None:
            raise ValueError("No object loaded. Call load_object first.")
            
        with torch.no_grad():
            contacts, _, _ = self.model.sample(self.current_object['verts'], 
                                             self.current_object['normals'])
        
        heatmap = contacts.squeeze().cpu().numpy()
        
        if visualize:
            pc = open3d.geometry.PointCloud()
            pc.points = open3d.utility.Vector3dVector(self.current_object['verts'][0].cpu().numpy())
            colors = np.zeros((len(self.current_object['verts'][0]), 3))
            colors[:, 0] = heatmap
            colors[:, 2] = 1 - heatmap
            pc.colors = open3d.utility.Vector3dVector(colors)
            open3d.visualization.draw_geometries([pc])
            
        return heatmap

def test_contact_predictors():
    # Initialize predictors
    contactdb_model = ContactDBHeuristic()
    contactgen_model = ContactGenHeuristic()

    # Load and preprocess test mesh
    mesh_path = "/home/bowenj/Projects/DexFun/reconstruction/mesh_raw/scissors.stl"
    mesh = trimesh.load(mesh_path)
    mesh.apply_scale(0.2)

    print("Testing ContactDB...")
    try:
        contactdb_model.load_object(mesh)
        db_heatmap = contactdb_model.query(visualize=True)
        print(f"ContactDB heatmap shape: {db_heatmap.shape}")
        print(f"ContactDB heatmap range: [{db_heatmap.min():.3f}, {db_heatmap.max():.3f}]")
        print("ContactDB test passed!")
    except Exception as e:
        print(f"ContactDB test failed with error: {e}")

    print("\nTesting ContactGen...")
    try:
        contactgen_model.load_object(mesh)
        cg_heatmap = contactgen_model.query(visualize=True)
        print(f"ContactGen heatmap shape: {cg_heatmap.shape}")
        print(f"ContactGen heatmap range: [{cg_heatmap.min():.3f}, {cg_heatmap.max():.3f}]")
        print("ContactGen test passed!")
    except Exception as e:
        print(f"ContactGen test failed with error: {e}")

if __name__ == "__main__":
    test_contact_predictors()