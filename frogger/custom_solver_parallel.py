import numpy as np
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count
import nlopt
from dataclasses import dataclass
from copy import deepcopy

from frogger.solvers import Frogger, FroggerConfig
from frogger.utils import timeout

@dataclass
class GraspAttemptResult:
    """Class to store the result of a single grasp generation attempt."""
    success: bool
    q_star: Optional[np.ndarray]
    q0: Optional[np.ndarray]
    error_msg: Optional[str]
    violations: Dict[str, float]

class FunctionalFrogger(Frogger):
    """Frogger solver with functional contact constraints and parallel grasp generation."""
    
    def __init__(
        self, 
        cfg: FroggerConfig,
        functional_contacts: List[Tuple[np.ndarray, Optional[np.ndarray]]]
    ) -> None:
        super().__init__(cfg)
        self.model.functional_contacts = functional_contacts
        
        # Store configuration for creating new instances
        self.cfg = cfg
        self.functional_contacts = functional_contacts

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

    def _attempt_grasp(self, optimize: bool, check_constraints: bool, **kwargs) -> GraspAttemptResult:
        """Single grasp attempt."""
        try:
            # Sample initial configuration
            try:
                q0, _ = timeout(60.0)(self.sampler.sample_configuration)(**kwargs)
            except RuntimeError as e:
                return GraspAttemptResult(
                    success=False,
                    q_star=None,
                    q0=None,
                    error_msg=f"Sampling error: {str(e)}",
                    violations={}
                )

            if not optimize:
                if check_constraints:
                    success = self.check_constraints(q0)
                    return GraspAttemptResult(
                        success=success,
                        q_star=None,
                        q0=q0,
                        error_msg=None if success else "Constraint check failed",
                        violations={}
                    )
                else:
                    return GraspAttemptResult(
                        success=True,
                        q_star=None,
                        q0=q0,
                        error_msg=None,
                        violations={}
                    )

            # Set current configuration to compute poses
            self.model.set_q(q0)
            
            # Compute correspondence and update model
            fingertip_poses = self.model.compute_fingertip_poses()
            self.model.contact_correspondence = self.compute_contact_correspondence(fingertip_poses)

            # Optimize
            try:
                q_star = self.opt.optimize(q0)
            except (RuntimeError, ValueError, nlopt.RoundoffLimited) as e:
                return GraspAttemptResult(
                    success=False,
                    q_star=None,
                    q0=q0,
                    error_msg=f"Optimization error: {str(e)}",
                    violations={}
                )

            # Check for NaN values
            if np.any(np.isnan(q_star)):
                return GraspAttemptResult(
                    success=False,
                    q_star=q_star,
                    q0=q0,
                    error_msg="Optimization returned NaN values",
                    violations={}
                )

            # Verify constraints
            g_val = np.zeros(self.n_ineq)
            self.g(g_val, q_star, np.zeros(0))
            h_val = np.zeros(self.n_eq)
            self.h(h_val, q_star, np.zeros(0))

            # Compute violations
            violations = {
                'surface': np.max(np.abs(h_val[: self.n_surf])),
                'coupling': (np.max(np.abs(h_val[self.n_surf : (self.n_surf + self.model.n_couple)]))
                           if self.model.n_couple > 0 else 0.0),
                'h_extra': (np.max(np.abs(h_val[(self.n_surf + self.model.n_couple):]))
                           if len(h_val[(self.n_surf + self.model.n_couple):]) > 0 else 0.0),
                'joint': max(np.max(g_val[: self.model.n_bounds]), 0.0),
                'collision': max(np.max(g_val[self.model.n_bounds : (self.model.n_bounds + self.model.n_pairs)]), 0.0),
                'g_extra': (max(np.max(g_val[(self.model.n_bounds + self.model.n_pairs):]), 0.0)
                           if len(g_val[(self.model.n_bounds + self.model.n_pairs):]) > 0 else 0.0)
            }

            # Check if solution is feasible
            success = (
                violations['surface'] <= self.tol_surf
                and violations['coupling'] <= self.tol_couple
                and violations['joint'] <= self.tol_joint
                and violations['collision'] <= self.tol_col
                and violations['g_extra'] <= self.tol_fclosure
                and violations['h_extra'] <= self.tol_fclosure
            )

            return GraspAttemptResult(
                success=success,
                q_star=q_star,
                q0=q0,
                error_msg=None if success else "Constraint violations",
                violations=violations
            )
            
        except Exception as e:
            return GraspAttemptResult(
                success=False,
                q_star=None,
                q0=None,
                error_msg=f"Unexpected error: {str(e)}",
                violations={}
            )

    def generate_grasp(
        self,
        optimize: bool = True,
        check_constraints: bool = False,
        n_processes: Optional[int] = None,
        max_attempts: int = 100,
        **kwargs
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate grasp with functional contacts using parallel processing.
        
        Parameters
        ----------
        optimize : bool, default=True
            Whether to optimize the sampled configuration
        check_constraints : bool, default=False
            Whether to check constraints for non-optimized samples
        n_processes : int, optional
            Number of parallel processes to use. If None, uses CPU count
        max_attempts : int, default=100
            Maximum number of attempts before giving up
        **kwargs
            Additional arguments passed to sample_configuration
        """
        if n_processes is None:
            n_processes = cpu_count()

        attempts_made = 0
        while attempts_made < max_attempts:
            # Create multiple independent instances for parallel processing
            batch_size = min(n_processes, max_attempts - attempts_made)
            frogger_instances = []
            
            for _ in range(batch_size):
                # Create a new instance with the same configuration
                new_cfg = deepcopy(self.cfg)
                new_frogger = FunctionalFrogger(new_cfg, self.functional_contacts)
                frogger_instances.append(new_frogger)
            
            # Try multiple grasp attempts in parallel
            with Pool(n_processes) as pool:
                results = pool.starmap(
                    FunctionalFrogger._attempt_grasp,
                    [(f, optimize, check_constraints, kwargs) for f in frogger_instances]
                )
            
            attempts_made += batch_size
            
            # Process results
            for result in results:
                if result.success:
                    print(f"Success: Found solution after {attempts_made} attempts")
                    return result.q_star, result.q0
                else:
                    if result.error_msg:
                        print(f"Failed attempt: {result.error_msg}")
                    if result.violations:
                        self._print_violations(result.violations)
            
            print(f"Batch failed. Made {attempts_made}/{max_attempts} attempts...")
        
        print(f"All {max_attempts} attempts failed. Giving up.")
        return None, None

    def _print_violations(self, violations: Dict[str, float]) -> None:
        """Print constraint violations."""
        if violations['surface'] > self.tol_surf:
            print(f"Surface contact violation: {violations['surface']:.2e} > {self.tol_surf}")
        if violations['coupling'] > self.tol_couple:
            print(f"Coupling violation: {violations['coupling']:.2e} > {self.tol_couple}")
        if violations['joint'] > self.tol_joint:
            print(f"Joint limit violation: {violations['joint']:.2e} > {self.tol_joint}")
        if violations['collision'] > self.tol_col:
            print(f"Collision violation: {violations['collision']:.2e} > {self.tol_col}")
        if violations['g_extra'] > self.tol_fclosure or violations['h_extra'] > self.tol_fclosure:
            print(f"Force closure violation: g:{violations['g_extra']:.2e}, h:{violations['h_extra']:.2e} > {self.tol_fclosure}")