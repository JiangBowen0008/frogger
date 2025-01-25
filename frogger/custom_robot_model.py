import numpy as np
from typing import Optional, List, Tuple
from pydrake.math import RigidTransform

from frogger.robots.robot_core import RobotModel, RobotModelConfig
from frogger.robots.robots import AlgrModel, AlgrModelConfig
from frogger.robots.custom_robots import LeapModel, LeapModelConfig
from frogger.grasping import (
    compute_gOCs,
    compute_grasp_matrix,
    compute_primitive_forces,
    wedge,
)
from frogger.metrics import min_weight_gradient, min_weight_lp

def get_contact_names(cfg: RobotModelConfig) -> list[str]:
    if isinstance(cfg, AlgrModelConfig):
        hand = cfg.hand
        return [
            f"algr_{hand}_if_ds",
            f"algr_{hand}_mf_ds",
            f"algr_{hand}_rf_ds",
            f"algr_{hand}_th_ds",
        ]
    elif isinstance(cfg, LeapModelConfig):
        return [
            f"fingertip",
            f"fingertip_2",
            f"fingertip_3",
            f"thumb_fingertip",
        ]
    
def get_offsets(cfg: RobotModelConfig) -> list[np.ndarray]:
    # Contact point offsets in the fingertip frame
    if isinstance(cfg, AlgrModelConfig):
        th_t = np.pi / 4.0
        r_f = 0.012
        contact_locs = [
            np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # index
            np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # middle
            np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # ring
            np.array([r_f * np.sin(th_t), 0.0, 0.0423 + r_f * np.cos(th_t)]),  # thumb
        ]
        return contact_locs
    else:
        contact_locs = [
            np.array([-0.011, -0.035, 0.01318]) for _ in range(3)
        ] + [np.array([-0.009, -0.048, -0.01128])]
        return contact_locs
    
class FunctionalRobotModel(AlgrModel):
    """Robot model that supports required contact points."""
    
    def __init__(self, cfg: RobotModelConfig) -> None:
        super().__init__(cfg)
        # Initialize attributes for functional contacts
        self.functional_contacts: Optional[List[Tuple[np.ndarray, Optional[np.ndarray]]]] = None 
        self.contact_correspondence: Optional[List[int]] = None
        self.functional_constraint_type: str = "fingertip"
        self.contact_names = get_contact_names(cfg)
        self.contact_locs = get_offsets(cfg)
    
    def compute_Df(self, q: np.ndarray) -> np.ndarray:
        df = super().compute_Df(q)
        df *= 1e-2
        return df

    def compute_fingertip_poses(self, finger_idx: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.plant_context is None:
            raise RuntimeError("Plant context not initialized")
            
        hand = self.hand
        # Get fingertip frames
        fingertip_frames = [
            self.plant.GetBodyByName(name).body_frame() for name in self.contact_names
        ]        

        if finger_idx is not None:
            frame = fingertip_frames[finger_idx]
            X_WF = self.plant.CalcRelativeTransform(
                self.plant_context,
                self.plant.world_frame(),
                frame
            )
            p_WC = self.plant.CalcPointsPositions(
                self.plant_context,
                frame,
                self.contact_locs[finger_idx],
                self.plant.world_frame(),
            ).squeeze()
            return [(p_WC, X_WF.rotation().matrix())]

        fingertip_poses = []
        for frame, contact_loc in zip(fingertip_frames, self.contact_locs):
            X_WF = self.plant.CalcRelativeTransform(
                self.plant_context,
                self.plant.world_frame(),
                frame
            )
            p_WC = self.plant.CalcPointsPositions(
                self.plant_context,
                frame,
                contact_loc,
                self.plant.world_frame(),
            ).squeeze()
            fingertip_poses.append((p_WC, X_WF.rotation().matrix()))
        return fingertip_poses
    
    def compute_contact_poses(self, q, finger_idx: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Compute contact poses by finding collision points."""
        if self.plant_context is None:
            raise RuntimeError("Plant context not initialized")
            
        # Process collisions to get contact points if not already computed
        if self.p_tips is None:
            self._process_collisions(self.plant.GetPositions(self.plant_context, self.robot_instance))

        contact_poses = [(pos, normal) for pos, normal in zip(self.p_tips, self.compute_n_W(q))]
        if finger_idx is not None:
            return [contact_poses[finger_idx]]
        
        return contact_poses
    
    # def compute_fingertip_poses(self, finger_idx: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    #     """Compute fingertip poses by finding collision points."""
    #     if self.plant_context is None:
    #         raise RuntimeError("Plant context not initialized")
            
    #     # Process collisions to get fingertip points
    #     if self.p_tips is None:
    #         self._process_collisions(self.plant.GetPositions(self.plant_context, self.robot_instance))
        
    #     if finger_idx is not None:
    #         return [(self.p_tips[finger_idx], self.compute_n_W(self.q)[finger_idx])]
            
    #     return [(pos, normal) for pos, normal in zip(self.p_tips, self.compute_n_W(self.q))]
    
    def compute_normals(self, pos):
        """Compute normals for the given positions."""
        return self.obj.Ds_W(pos, batched=True)
    
    def _compute_G_and_W(self) -> None:
        """Computes grasp and wrench matrices, with minimal contribution from functional contacts."""
        # Basic computations same as original 
        X_WO = self.obj.X_WO
        X_OW = X_WO.inverse()
        R_OW = X_OW.inverse().rotation().matrix()
        
        self.P_OF = X_OW @ self.p_tips.T  # (3, nc)
        Ds_p = self.obj.Ds_W(self.p_tips, batched=True)
        self.n_W = -Ds_p.T / np.linalg.norm(Ds_p, axis=1)
        self.n_O = R_OW @ self.n_W

        # Compute full matrices first
        self.gOCs = compute_gOCs(self.P_OF, self.n_O)
        self.G = compute_grasp_matrix(self.gOCs)
        self.W = self.G @ np.kron(np.eye(self.nc), self.F)

        # Scale down contribution of functional contacts in W matrix
        if self.contact_correspondence is not None:
            for i in range(self.nc):
                if self.contact_correspondence[i] != -1:
                    # Scale the columns corresponding to this contact to be very small
                    col_start = i * len(self.F) + 7
                    col_end = (i + 1) * len(self.F) + 7
                    self.W[:, col_start:col_end] *= 1e-3

    def _compute_l(self) -> None:
        """Computes the min-weight metric considering only grasp contacts."""
        super()._compute_l()
        l, dl = self.l, self.Dl
        for i in range(self.nc):
            if self.contact_correspondence[i] != -1:
                # Scale the columns corresponding to this contact to be very small
                col_start = i * len(self.F) + 7
                col_end = (i + 1) * len(self.F) + 7
                # l[col_start:col_end] *= 1e-3
                dl[col_start:col_end] *= 1e-3

  
    def _compute_eq_cons(self, q: np.ndarray) -> None:
        if self.h is None:
            self._init_eq_cons()
            
        # Pre-compute poses
        # Get positions based on mode
        # if self.functional_constraint_type == "contact":
        #     current_poses = self.compute_contact_poses()
        # elif self.functional_constraint_type == "fingertip":
        current_poses = self.compute_fingertip_poses()
        
        # For each finger, we'll have one scalar constraint
        h_constraints = np.zeros(self.nc)
        Dh_constraints = np.zeros((self.nc, self.n))
        
        max_distance = -np.inf
        for i in range(self.nc):
            if self.contact_correspondence and self.contact_correspondence[i] != -1:
                func_pos, func_dir = self.functional_contacts[self.contact_correspondence[i]]
                current_pos, current_rot = current_poses[i]
                
                # Signed distance
                pos_error = np.linalg.norm(current_pos - func_pos)
                # determine the sign
                sign = np.sign(np.dot(current_pos - func_pos, func_dir))
                pos_error *= sign

                max_distance = max(max_distance, pos_error)
                h_constraints[i] = pos_error
                # h_constraints[i] = self.h_tip[i]
                # print(self.h_tip[i])
                error_direction = (current_pos - func_pos) / pos_error
                # error_direction = (func_pos - current_pos) / pos_error
                # if pos_error > 1e-10:
                #     error_direction = (current_pos - func_pos) / pos_error
                # else:
                #     error_direction = np.zeros(3)
                Dh_constraints[i] = error_direction @ self.J_tips[i]
                # Dh_constraints[i] = 0.8 * error_direction @ self.J_tips[i] + 0.2 * self.Dh_tip[i]
                
                # Add direction err
                # if False:
                # if func_dir is not None:
                #     # Get current direction and compute angle
                #     current_dir = current_rot[:, 2]
                #     cos_theta = np.clip(np.dot(current_dir, func_dir), -1.0, 1.0)
                #     angle_error = np.arccos(cos_theta)
                    
                #     # Combine position and scaled angle error
                #     # Scale factor (5e-3) converts radians to approximate distance units
                #     h_constraints[i] += angle_error * 5e-3
                    
                #     # Update gradient for angle error when not perfectly aligned
                #     if cos_theta < 0.999:  # Avoid numerical issues near alignment
                #         rotation_axis = np.cross(current_dir, func_dir)
                #         rotation_axis_norm = np.linalg.norm(rotation_axis)
                        
                #         if rotation_axis_norm > 1e-10:  # Only update if rotation axis is well-defined
                #             # Normalize rotation axis
                #             rotation_axis = rotation_axis / rotation_axis_norm
                            
                #             # Compute angle gradient 
                #             # The factor -1/sqrt(1-cos_theta^2) comes from derivative of arccos
                #             # We multiply by 5e-3 to match the scaling in the constraint
                #             angle_grad_scale = -5e-3 / np.sqrt(1.0 - cos_theta**2)
                #             dir_grad = angle_grad_scale * (rotation_axis @ self.J_tips[i])
                            
                #             # Add direction gradient to position gradient
                #             Dh_constraints[i] += dir_grad
            else:
                # h_constraints[i] = self.h_tip[i]
                # Dh_constraints[i] = self.Dh_tip[i]
                h_constraints[i] = self.h_tip[i]
                Dh_constraints[i] = self.Dh_tip[i]
        # print(f"h constraints: {h_constraints}")
        # print(f"Dh constraints: {Dh_constraints}")
        # print(f"Max distance: {max_distance}")
        
        # Store constraints
        if self.n_couple != 0:
            h_couple = self.A_couple @ q + self.b_couple
            self.h[: self.n_couple + self.nc] = np.concatenate((h_constraints, h_couple))
            self.Dh[: self.n_couple + self.nc, :] = np.concatenate((Dh_constraints, self.A_couple))
        else:
            self.h[: self.nc] = h_constraints
            self.Dh[: self.nc, :] = Dh_constraints
        
        # Add any extra constraints
        if self.custom_compute_h is not None:
            h_extra, Dh_extra = self.custom_compute_h(self)
            assert len(h_extra) == self.n_h_extra
            self.h[self.n_couple + self.nc :] = h_extra
            self.Dh[self.n_couple + self.nc :, :] = Dh_extra