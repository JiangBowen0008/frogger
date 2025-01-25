from frogger.sampling import HeuristicICSampler
from frogger.robots.robots import AlgrModel, RobotModel
from frogger.robots.custom_robots import LeapModel

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.geometry import Sphere, Rgba


class FnHeuristicAlgrICSampler(HeuristicICSampler):
    """
    Heuristic sampler for the Allegro hand that also considers functional contact points.
    The logic:
    - If there is a first functional contact, assign it to the index finger (finger 0).
    - If there is a second functional contact, assign it to the thumb (finger 3).
    - Any additional functional contacts beyond the first two will cause an error 
      (or could be ignored, depending on desired behavior).
    - All other fingers that do not have a functional contact point get the default constraints.
    """

    def __init__(
        self, 
        model: AlgrModel, 
        functional_contacts: list[tuple[np.ndarray, np.ndarray|None]],
        table: bool = False, 
        z_axis_fwd: bool = False,
    ) -> None:
        super().__init__(model, table=table, z_axis_fwd=z_axis_fwd)
        self.functional_contacts = functional_contacts
        self.contact_names = model.contact_names

    def add_visualization(self, model, path_prefix="sampled_grasp"):
        """
        Visualize contact points with the following color scheme:
        - contacts:
            - Red: index finger (0)
            - Green: middle finger (1)
            - Blue: ring finger (2)
            - Yellow: thumb (3)
        """
        # Delete any existing visualization
        try:
            model.meshcat.Delete(path=path_prefix)
        except:
            pass

        # Visualize functional contacts with color-coded spheres
        for i, (pos, _) in self.func_assignments.items():
            sphere_name = f"{path_prefix}/contact_{i}"
            color = {
                0: Rgba(1.0, 0.0, 0.0, 1.0),  # Red for index
                1: Rgba(0.0, 1.0, 0.0, 1.0),  # Green for middle
                2: Rgba(0.0, 0.0, 1.0, 1.0),  # Blue for ring
                3: Rgba(1.0, 1.0, 0.0, 1.0),  # Yellow for thumb
            }[i]
            
            model.meshcat.SetObject(
                path=sphere_name,
                shape=Sphere(0.01),
                rgba=color
            )
            model.meshcat.SetTransform(
                path=sphere_name,
                X_ParentPath=RigidTransform(pos)
            )

    def add_additional_constraints(
        self, 
        ik: InverseKinematics, 
        X_WPalm_des: RigidTransform
    ) -> None:
        normal_tol_pos = 0.1
        # fn_tol_rot = 1.5
        fn_tol_rot = 0.8

        # 1. Set a default hand configuration guess
        # q_imr = np.array([0.0, 0.6, 0.6, 0.1])  # index/middle/ring default guess
        # q_th = np.array([1, 1.16, 0.6, 0.6])   # thumb default guess
        # q_hand = np.concatenate((q_imr, q_imr, q_imr, q_th))
        q_hand = np.zeros(16)
        q_curr = self.model.plant.GetPositions(self.model.plant_context)
        q_curr[-16:] = q_hand
        self.model.plant.SetPositions(self.model.plant_context, q_curr)

        # 2. Compute width w based on object orientation and palm alignment
        obj = self.model.obj
        X_WO = obj.X_WO
        X_OBB = obj.X_OBB
        R_OBB = X_OBB.rotation()
        R_WBB = X_WO.rotation() @ R_OBB
        axis_lengths_O = obj.ub_oriented - obj.lb_oriented

        z_pc = X_WPalm_des.rotation().matrix()[:, 2]  # palm's z-axis
        z_alignment = np.argmax(np.abs(R_WBB.inverse() @ z_pc))
        w = axis_lengths_O[z_alignment]

        x_pc = X_WPalm_des.rotation().matrix()[:, 0]  # palm's x-axis
        x_alignment = np.argmax(np.abs(R_WBB.inverse() @ x_pc))
        if x_alignment == 2:
            f_ext = min(X_WPalm_des.translation()[-1], 0.08)
        else:
            f_ext = 0.08

        # 3. Compute normal fingertip target positions
        hand = self.model.hand
        if hand == "rh":
            p_if = X_WPalm_des @ np.array([f_ext, 0.04, w / 2])   # index
            p_mf = X_WPalm_des @ np.array([f_ext, 0.0, w / 2])    # middle
            p_rf = X_WPalm_des @ np.array([f_ext, -0.04, w / 2])  # ring
            p_th = X_WPalm_des @ np.array([f_ext, 0.02, -w / 2])  # thumb
        else:
            p_if = X_WPalm_des @ np.array([f_ext, -0.04, w / 2])
            p_mf = X_WPalm_des @ np.array([f_ext, 0.0, w / 2])
            p_rf = X_WPalm_des @ np.array([f_ext, 0.04, w / 2])
            p_th = X_WPalm_des @ np.array([f_ext, -0.02, -w / 2])

        P_WFs = np.stack((p_if, p_mf, p_rf, p_th))

        # 4. Get the fingertip frames
        contact_bodies = [
            self.model.plant.GetBodyByName(name) for name in self.contact_names
        ]
        contact_frames = [b.body_frame() for b in contact_bodies]

        # Helper function to make rotation from direction
        def make_rotation_for_direction(d_vec: np.ndarray) -> RotationMatrix:
            # d_vec should be the surface normal direction
            x_unit = d_vec / np.linalg.norm(d_vec)
            
            # Add random rotation around x-axis
            theta = np.random.uniform(-np.pi, np.pi)  # Random angle
            
            # Choose initial reference direction
            ref = np.array([0, 1, 0]) if abs(x_unit[2]) > 0.9 else np.array([0, 0, 1])
            
            # Compute y-axis
            y_new = np.cross(x_unit, ref)  # Order changed to maintain right-hand rule
            y_new /= np.linalg.norm(y_new)
            
            # Compute z-axis 
            z_new = np.cross(x_unit, y_new)  # Order changed to maintain right-hand rule
            z_new /= np.linalg.norm(z_new)
            
            # Apply random rotation around x-axis
            c, s = np.cos(theta), np.sin(theta)
            y_new_rot = c * y_new + s * z_new
            z_new_rot = -s * y_new + c * z_new
            
            # Return rotation matrix with randomized y and z axes
            return RotationMatrix(np.column_stack((x_unit, y_new_rot, z_new_rot)))
        
        # def make_rotation_for_direction(d_vec: np.ndarray) -> RotationMatrix:
        #     # d_vec should be the surface normal direction
        #     z_unit = d_vec / np.linalg.norm(d_vec)
            
        #     # Add random rotation around z-axis
        #     theta = np.random.uniform(-np.pi, np.pi)  # Random angle
            
        #     # Choose initial reference direction
        #     ref = np.array([1, 0, 0]) if abs(z_unit[1]) > 0.9 else np.array([0, 1, 0])
            
        #     # Compute x-axis
        #     x_new = np.cross(ref, z_unit)  # Order changed to maintain right-hand rule
        #     x_new /= np.linalg.norm(x_new)
            
        #     # Compute y-axis
        #     y_new = np.cross(z_unit, x_new)  # Order changed to maintain right-hand rule
        #     y_new /= np.linalg.norm(y_new)
            
        #     # Apply random rotation around z-axis
        #     c, s = np.cos(theta), np.sin(theta)
        #     x_new_rot = c * x_new - s * y_new
        #     y_new_rot = s * x_new + c * y_new
            
        #     # Return rotation matrix with randomized x and y axes
        #     return RotationMatrix(np.column_stack((x_new_rot, y_new_rot, z_unit)))
                
        # 6. Apply functional contact logic:
        # - If one functional contact: assign to index finger (0)
        # - If two functional contacts: second one goes to thumb (3)
        # - If more than two: raise an error
        num_func = len(self.functional_contacts)
        # if num_func > 2:
        #     raise ValueError("More than two functional contacts are not supported.")

        # Assign functional contacts
        func_assignments = {}
        if num_func == 4:
            # All functional contacts are present
            func_assignments = {i: self.functional_contacts[i] for i in range(4)}
            # print("All functional contacts are present")
        elif num_func > 0:
            # rnd_f = np.random.choice([0,3])
            # func_assignments[rnd_f] = self.functional_contacts[0]
            # # First functional contact to index finger (0) or thumb (3)
            # # Second functional contact to the other
            # if num_func > 1:
            #     rnd_f2 = 0 if rnd_f == 3 else 3
            #     func_assignments[rnd_f2] = self.functional_contacts[1]
            # if num_func > 2:
            #     rnd_f3 = 0 if rnd_f == 3 or rnd_f2 == 3 else 3
            #     func_assignments[rnd_f3] = self.functional_contacts[2]
            func_assignments[0] = self.functional_contacts[0]
            if num_func > 1:
                # Randomly assign the rest of the fingers to the rest of the functional contacts (non repeating)
                options = [1, 2, 3]
                for i in range(1, num_func):
                    rnd_f = np.random.choice(options)
                    options.remove(rnd_f)
                    func_assignments[rnd_f] = self.functional_contacts[i]
        self.func_assignments = func_assignments

        # Add constraints
        for i in range(4):
            if i in func_assignments:
                # This finger has a functional contact
                # func_pos, func_dir = func_assignments[i]
                # # Position constraint
                # ik.AddPositionConstraint(
                #     contact_frames[i],
                #     self.model.contact_locs[i],
                #     self.model.plant.world_frame(),
                #     func_pos - 1e-4,
                #     func_pos + 1e-4,
                # )
                # # Orientation constraint if func_dir is given
                # if func_dir is not None:
                #     if func_dir.shape == (3,):
                #         R_des = make_rotation_for_direction(-func_dir)
                #     else:
                #         R_des = RotationMatrix(func_dir.copy())
                #     ik.AddOrientationConstraint(
                #         contact_frames[i],
                #         RotationMatrix(),  # finger frame to finger frame
                #         self.model.plant.world_frame(),
                #         R_des,
                #         fn_tol_rot  # orientation tolerance
                #     )
                # break
                pass
            else:
                # Normal fingertip position constraint
                # ik.AddPositionConstraint(
                #     contact_frames[i],
                #     contact_locs[i],
                #     self.model.plant.world_frame(),
                #     P_WFs[i, :] - normal_tol_pos,
                #     P_WFs[i, :] + normal_tol_pos,
                # )
                # No orientation constraint for normal contacts
                pass

class PalmHeuristicAlgrICSampler(FnHeuristicAlgrICSampler):
    def __init__(self, 
                 model: AlgrModel, 
                 functional_contacts: list[tuple[np.ndarray, np.ndarray|None]],
                 contact_predictor,
                 table: bool = False, 
                 z_axis_fwd: bool = False,
                 palm_offset: float = 0.05
                ) -> None:
        super().__init__(model, functional_contacts, table=table, z_axis_fwd=z_axis_fwd)
        self.contact_predictor = contact_predictor
        self.palm_offset = palm_offset

    def sample_palm_pose(self) -> RigidTransform:
        obj = self.model.obj
        mesh = obj.mesh
        if getattr(self.contact_predictor, 'current_object') is None:
            self.contact_predictor.load_object(mesh)
        heatmap, normals = self.contact_predictor.query()
        vertices = np.array(mesh.vertices)

        # Sample contact point
        contact_idx = np.random.choice(len(heatmap), p=heatmap/heatmap.sum())
        contact_point_O = vertices[contact_idx]
        contact_point_W = obj.X_WO @ contact_point_O
        
        # Get corresponding normal
        normal_O = normals[contact_idx]
        normal_W = obj.X_WO.rotation().matrix() @ normal_O
        normal_W = normal_W / np.linalg.norm(normal_W)

        # Set x-axis opposite to surface normal (palm facing contact)
        x_hat = -normal_W

        # Select y-axis similar to original sampler
        X_WO = obj.X_WO
        X_OBB = obj.X_OBB
        axis_lengths_O = obj.ub_oriented - obj.lb_oriented
        
        prob_1 = axis_lengths_O[0] / np.sum(axis_lengths_O)
        prob_2 = axis_lengths_O[1] / np.sum(axis_lengths_O)
        num = np.random.rand()
        if num <= prob_1:
            hand_y_axis_ind = 0
        elif num <= prob_1 + prob_2:
            hand_y_axis_ind = 1
        else:
            hand_y_axis_ind = 2

        hand_y_axis_O = np.zeros(3)
        hand_y_axis_O[hand_y_axis_ind] = 1.0
        hand_y_axis_W = (X_WO @ X_OBB).rotation().matrix() @ hand_y_axis_O

        # Add randomization to y-axis
        rand_sign = 1 if np.random.random() < 0.5 else -1
        _y_hat = rand_sign * hand_y_axis_W
        _y_hat = _y_hat / np.linalg.norm(_y_hat)
        y_hat = np.random.vonmises(mu=_y_hat, kappa=20)
        y_hat = y_hat / np.linalg.norm(y_hat)
        
        # Project onto plane perpendicular to x_hat and normalize
        y_hat = y_hat - (np.dot(y_hat, x_hat) * x_hat)
        y_hat = y_hat / np.linalg.norm(y_hat)

        # Complete right-handed frame
        z_hat = np.cross(x_hat, y_hat)

        if self.z_axis_fwd and z_hat[0] < 0:
            y_hat, z_hat = -y_hat, -z_hat

        # Form pose
        R_WPalm = np.stack((x_hat, y_hat, z_hat), axis=1)
        P_WPalm = contact_point_W + self.palm_offset * -x_hat

        return RigidTransform(RotationMatrix(R_WPalm), P_WPalm)


class ContactHeuristicAlgrICSampler(FnHeuristicAlgrICSampler):
    def __init__(self, 
                 model: AlgrModel, 
                 functional_contacts: list[tuple[np.ndarray, np.ndarray|None]],
                 contact_predictor,
                 table: bool = False, 
                 z_axis_fwd: bool = False,
                 finger_spacing: float = 0.03  # Approximate desired spacing between fingers
                ) -> None:
        super().__init__(model, functional_contacts, table=table, z_axis_fwd=z_axis_fwd)
        self.contact_predictor = contact_predictor
        self.finger_spacing = finger_spacing
    
    def add_visualization(self, model, path_prefix="sampled_grasp"):
        """
        Visualize contact points with the following color scheme:
        - Black: functional contacts
        - Non-functional contacts:
            - Red: index finger (0)
            - Green: middle finger (1)
            - Blue: ring finger (2)
            - Yellow: thumb (3)
        """
        # Delete any existing visualization
        try:
            model.meshcat.Delete(path=path_prefix)
        except:
            pass

        # Color mapping for non-functional fingers
        finger_colors = {
            0: Rgba(1.0, 0.0, 0.0, 1.0),  # Red for index
            1: Rgba(0.0, 1.0, 0.0, 1.0),  # Green for middle
            2: Rgba(0.0, 0.0, 1.0, 1.0),  # Blue for ring
            3: Rgba(1.0, 1.0, 0.0, 1.0),  # Yellow for thumb
        }

        # Visualize functional contacts (black)
        for i, (pos, _) in self.func_assignments.items():
            sphere_name = f"{path_prefix}/functional_contact_{i}"
            model.meshcat.SetObject(
                path=sphere_name,
                shape=Sphere(0.01),
                rgba=Rgba(0.0, 0.0, 0.0, 1.0)  # Black
            )
            model.meshcat.SetTransform(
                path=sphere_name,
                X_ParentPath=RigidTransform(pos)
            )

        # Visualize normal contacts (color-coded by finger)
        obj = model.obj
        for i, pos_local in self.normal_assignments.items():
            pos_world = obj.X_WO @ pos_local
            sphere_name = f"{path_prefix}/normal_contact_{i}"
            model.meshcat.SetObject(
                path=sphere_name,
                shape=Sphere(0.01),
                rgba=finger_colors[i]  # Color based on finger index
            )
            model.meshcat.SetTransform(
                path=sphere_name,
                X_ParentPath=RigidTransform(pos_world)
            )

    def _get_point_y_in_palm(self, point: np.ndarray, X_WPalm_des: RigidTransform) -> float:
        """Convert point to palm frame and get y coordinate"""
        X_PW = X_WPalm_des.inverse()
        p_P = X_PW @ point
        return p_P[1]

    def _sample_points_with_spacing(self, 
                                  vertices: np.ndarray, 
                                  heatmap: np.ndarray,
                                  X_WPalm_des: RigidTransform,
                                  n_points: int,
                                  existing_points: list[np.ndarray] = None) -> list[np.ndarray]:
        """Sample points with approximate spacing in palm y direction"""
        if existing_points is None:
            existing_points = []
            
        # Convert all vertices to palm frame y-coordinates for reference
        y_coords = np.array([self._get_point_y_in_palm(v, X_WPalm_des) for v in vertices])
        
        # Get existing y-coordinates
        existing_y = [self._get_point_y_in_palm(p, X_WPalm_des) for p in existing_points]
        
        sampled_points = []
        for _ in range(n_points):
            # Sample candidates
            n_candidates = 10  # Number of candidates to consider each time
            candidates_idx = np.random.choice(
                len(heatmap), 
                size=n_candidates, 
                p=heatmap/heatmap.sum()
            )
            candidates = vertices[candidates_idx]
            candidates_y = y_coords[candidates_idx]
            
            # Score candidates based on spacing from existing points
            scores = []
            for y in candidates_y:
                # Calculate minimum distance to any existing point
                if not existing_y and not sampled_points:
                    min_dist = float('inf')
                else:
                    all_y = existing_y + [self._get_point_y_in_palm(p, X_WPalm_des) for p in sampled_points]
                    min_dist = min(abs(y - ey) for ey in all_y) if all_y else float('inf')
                
                # Score is a combination of spacing and some randomness
                spacing_score = 1.0 if min_dist >= self.finger_spacing else min_dist/self.finger_spacing
                random_factor = np.random.uniform(0.8, 1.2)  # Add some randomness
                scores.append(spacing_score * random_factor)
            
            # Select the best candidate
            best_idx = np.argmax(scores)
            sampled_points.append(candidates[best_idx])
            
        return sampled_points

    def _assign_points_to_fingers(self, 
                                points: list[np.ndarray], 
                                X_WPalm_des: RigidTransform,
                                available_fingers: list[int]) -> dict:
        """Assign points to fingers based on palm y-coordinate"""
        # Get y-coordinate for each point in palm frame
        point_ys = [self._get_point_y_in_palm(p, X_WPalm_des) for p in points]
        
        # Sort points and fingers by y-coordinate
        sorted_pairs = sorted(zip(point_ys, points))
        if self.model.hand == "rh":
            # For right hand: index(0) -> ring(2) is positive to negative y
            sorted_fingers = sorted(available_fingers, reverse=True)
        else:
            # For left hand: index(0) -> ring(2) is negative to positive y
            sorted_fingers = sorted(available_fingers)
            
        return {f: p for f, (_, p) in zip(sorted_fingers, sorted_pairs)}

    def add_additional_constraints(self, 
                             ik: InverseKinematics, 
                             X_WPalm_des: RigidTransform) -> None:
        # First call parent's add_additional_constraints to setup func_assignments and other initial stuff
        super().add_additional_constraints(ik, X_WPalm_des)
        
        # Now we can use self.func_assignments which was set by the parent class
        normal_tol_pos = 0.01
        fn_tol_rot = 0.5

        # Get frames and contact locations
        hand = self.model.hand
        contact_bodies = [
            self.model.plant.GetBodyByName(name) for name in self.model.contact_names
        ]
        contact_frames = [b.body_frame() for b in contact_bodies]

        th_t = np.pi / 4.0 + (np.pi / 24.0) * np.random.randn()
        r_f = 0.012
        contact_locs = self.model.contact_locs

        # Get contact predictions
        obj = self.model.obj
        mesh = obj.mesh
        if getattr(self.contact_predictor, 'current_object') is None:
            self.contact_predictor.load_object(mesh)
        heatmap = self.contact_predictor.query()
        
        if heatmap.max() > 1.0:
            heatmap = heatmap / heatmap.max()
            
        vertices = np.array(mesh.vertices)
        
        # Get functional contact points for spacing consideration
        functional_points = [self.func_assignments[i][0] for i in self.func_assignments]
        
        # Get available fingers for normal contacts
        available_fingers = [i for i in range(4) if i not in self.func_assignments]
        
        # Sample points with spacing consideration
        sampled_points = self._sample_points_with_spacing(
            vertices, 
            heatmap, 
            X_WPalm_des, 
            len(available_fingers),
            functional_points
        )
        
        # Assign points to fingers
        normal_assignments = self._assign_points_to_fingers(
            sampled_points, 
            X_WPalm_des, 
            available_fingers
        )
        
        # Store normal assignments for visualization
        self.normal_assignments = normal_assignments
        
        # Add constraints for normal contacts only (functional ones were added by parent)
        for i in range(4):
            if i not in self.func_assignments:
                p_W = obj.X_WO @ normal_assignments[i]
                # p_W = normal_assignments[i]
                ik.AddPositionConstraint(
                    contact_frames[i],
                    contact_locs[i],
                    self.model.plant.world_frame(),
                    p_W - normal_tol_pos,
                    p_W + normal_tol_pos,
                )
                # break
        
        # self.add_visualization()