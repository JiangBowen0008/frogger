import numpy as np
from typing import List, Tuple, Optional

from frogger.solvers import Frogger, FroggerConfig
from frogger.utils import timeout

import nlopt

from frogger.custom_robot_model import FunctionalRobotModel

class FunctionalFrogger(Frogger):
    """Frogger solver with functional contact constraints."""
    
    def __init__(
        self, 
        cfg: FroggerConfig,
        functional_contacts: List[Tuple[np.ndarray, Optional[np.ndarray]]]
    ) -> None:
        # if not isinstance(cfg.model, FunctionalRobotModel):
        #     raise TypeError("Model must be FunctionalRobotModel")
        
        super().__init__(cfg)
        self.model.functional_contacts = functional_contacts

    def compute_contact_correspondence(self, fingertip_poses: List[Tuple[np.ndarray, np.ndarray]]) -> List[int]:
        """Compute correspondence between fingertips and functional contacts."""
        fingertip_positions = np.array([pose[0] for pose in fingertip_poses])
        n_fingers = len(fingertip_poses)
        n_functional = len(self.model.functional_contacts)
        
        # Compute distance matrix
        distances = np.zeros((n_fingers, n_functional))
        for i, (pos, _) in enumerate(fingertip_poses):
            for j, (func_pos, _) in enumerate(self.model.functional_contacts):
                distances[i, j] = np.linalg.norm(pos - func_pos)
        
        # Assign correspondences
        correspondence = [-1] * n_fingers
        assigned = set()
        
        for _ in range(n_functional):
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            if j not in assigned:
                correspondence[i] = j
                assigned.add(j)
            distances[i, :] = np.inf
            distances[:, j] = np.inf
            
        return correspondence

    def generate_grasp(self, optimize=True, check_constraints=False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate grasp with functional contacts."""
        success = False
        while not success:
            # Sample initial configuration
            # q0, _ = timeout(60.0)(self.sampler.sample_configuration)(**kwargs)
            q0, _ = self.sampler.sample_configuration(**kwargs)
            # try:
            #     q0, _ = timeout(60.0)(self.sampler.sample_configuration)(**kwargs)
            # except RuntimeError as e:
            #     print(f"Sampling error: {e}. Resampling....")
            #     continue

             # Compute correspondence and update model
            self.model.set_q(q0)
            fingertip_poses = self.model.compute_fingertip_poses()
            self.model.contact_correspondence = self.compute_contact_correspondence(fingertip_poses)
            
            if not optimize:
                if check_constraints:
                    success = self.check_constraints(q0)
                    if success:
                        return None, q0
                    else:
                        continue
                else:
                    return None, q0            

            # Optimize
            # q_star = self.opt.optimize(q0)
            try:
                q_star = self.opt.optimize(q0)
            except (RuntimeError, ValueError, nlopt.RoundoffLimited) as e:
                print(f"Optimization error: {e}! Resampling...")
                q_star = np.nan * np.ones(self.model.n)
                continue

            # Check solution
            if np.any(np.isnan(q_star)):
                print("Failed: Optimization returned NaN values")
                continue
                
            # Verify constraints
            g_val = np.zeros(self.n_ineq)
            self.g(g_val, q_star, np.zeros(0))
            h_val = np.zeros(self.n_eq)
            self.h(h_val, q_star, np.zeros(0))

            # Check constraint violations
            surf_vio = np.max(np.abs(h_val[: self.n_surf]))
            couple_vio = np.max(np.abs(h_val[self.n_surf : (self.n_surf + self.n_couple)])) if self.model.n_couple > 0 else 0.0
            h_extra_vio = np.max(np.abs(h_val[(self.n_surf + self.n_couple):])) if len(h_val[(self.n_surf + self.n_couple):]) > 0 else 0.0
            
            joint_vio = max(np.max(g_val[: self.model.n_bounds]), 0.0)
            col_vio = max(np.max(g_val[self.model.n_bounds : (self.model.n_bounds + self.model.n_pairs)]), 0.0)
            g_extra_vio = max(np.max(g_val[(self.model.n_bounds + self.model.n_pairs):]), 0.0) if len(g_val[(self.model.n_bounds + self.model.n_pairs):]) > 0 else 0.0

            # Print specific failure reasons
            if surf_vio > self.tol_surf:
                print(f"Failed: Surface contact constraint violation ({surf_vio:.2e} > {self.tol_surf})")
            if couple_vio > self.tol_couple:
                print(f"Failed: Coupling constraint violation ({couple_vio:.2e} > {self.tol_couple})")
            if joint_vio > self.tol_joint:
                print(f"Failed: Joint limit violation ({joint_vio:.2e} > {self.tol_joint})")
            if col_vio > self.tol_col:
                print(f"Failed: Collision constraint violation ({col_vio:.2e} > {self.tol_col})")
            if g_extra_vio > self.tol_fclosure or h_extra_vio > self.tol_fclosure:
                print(f"Failed: Force closure constraint violation (g:{g_extra_vio:.2e}, h:{h_extra_vio:.2e} > {self.tol_fclosure})")

            # Check if solution is feasible
            success = (
                surf_vio <= self.tol_surf
                and couple_vio <= self.tol_couple
                and joint_vio <= self.tol_joint
                and col_vio <= self.tol_col
                and g_extra_vio <= self.tol_fclosure
                and h_extra_vio <= self.tol_fclosure
            )
            
            if success:
                print("Success: All constraints satisfied")
                return q_star, q0
            

    def check_constraints(self, q):
        self.model.set_q(q)
        
        # Verify constraints
        g_val = np.zeros(self.n_ineq)
        self.g(g_val, q, np.zeros(0))
        h_val = np.zeros(self.n_eq)
        self.h(h_val, q, np.zeros(0))

        # Check constraint violations
        surf_vio = np.max(np.abs(h_val[: self.n_surf]))
        couple_vio = np.max(np.abs(h_val[self.n_surf : (self.n_surf + self.n_couple)])) if self.model.n_couple > 0 else 0.0
        h_extra_vio = np.max(np.abs(h_val[(self.n_surf + self.n_couple):])) if len(h_val[(self.n_surf + self.n_couple):]) > 0 else 0.0
        
        joint_vio = max(np.max(g_val[: self.model.n_bounds]), 0.0)
        col_vio = max(np.max(g_val[self.model.n_bounds : (self.model.n_bounds + self.model.n_pairs)]), 0.0)
        g_extra_vio = max(np.max(g_val[(self.model.n_bounds + self.model.n_pairs):]), 0.0) if len(g_val[(self.model.n_bounds + self.model.n_pairs):]) > 0 else 0.0

        # Print specific failure reasons
        if surf_vio > self.tol_surf:
            print(f"Failed: Surface contact constraint violation ({surf_vio:.2e} > {self.tol_surf})")
        if couple_vio > self.tol_couple:
            print(f"Failed: Coupling constraint violation ({couple_vio:.2e} > {self.tol_couple})")
        if joint_vio > self.tol_joint:
            print(f"Failed: Joint limit violation ({joint_vio:.2e} > {self.tol_joint})")
        if col_vio > self.tol_col:
            print(f"Failed: Collision constraint violation ({col_vio:.2e} > {self.tol_col})")
        if g_extra_vio > self.tol_fclosure or h_extra_vio > self.tol_fclosure:
            print(f"Failed: Force closure constraint violation (g:{g_extra_vio:.2e}, h:{h_extra_vio:.2e} > {self.tol_fclosure})")

        # Check if solution is feasible
        success = (
            surf_vio <= self.tol_surf
            and couple_vio <= self.tol_couple
            and joint_vio <= self.tol_joint
            and col_vio <= self.tol_col
            and g_extra_vio <= self.tol_fclosure
            and h_extra_vio <= self.tol_fclosure
        )
