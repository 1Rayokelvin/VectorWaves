"""
Topological Singularity Detection
==================================

Find and analyze polarization singularities in 3D electromagnetic fields.
"""

import numpy as np, warnings
from .engine_stuff import FieldEngine

class SingularityFinder:
    """
    Locates and traces electric field singularities in 3D space.
    
    Works with a physics_engine to find points where field properties 
    become degenerate (C-points, C^T-points, L^T-points) and trace their evolution 
    through space as lines or loops.
    
    Time t=0 by default as polarization is time independent in monochromatic fields. 
    Use self.t to change this behaviour.
    
    Parameters
    ----------
    physics_engine : FieldEngine
        An initialized FieldEngine capable of evaluating points in space.
    
    Example
    -------
    ```python
    engine = FieldEngine(beam, cfg)
    fields = engine.compute_on_op(z=0.0)
    
    finder = SingularityFinder(engine)
    pts = finder.find_stokes_C_points(z_value=0.0, E_grid=fields.E)
    lines = finder.trace_stokes_C_lines(pts, ds=0.05)
    ```
    """
    def __init__(self, physics_engine: FieldEngine):
        self.engine = physics_engine
        
        # Read the engine's active backend to issue helpful warnings
        self.backend_name = self.engine.backend_name
        if self.backend_name == 'numpy':
            warnings.warn(
                "Singularity finder is using the 'numpy' backend. "
                "Use 'numba' or 'cupy64' for a significant speed boost.",
                RuntimeWarning
            )

        elif self.backend_name == 'cupy32':
            raise RuntimeError(
                "cupy32 backend is not supported: convergence heuristics assume float64 precision. "
                "Use 'cupy64'."
            )

        self.x = physics_engine.x
        self.y = physics_engine.y
        self.t = 0 
        self.x_min, self.x_max = min(self.x), max(self.x)
        self.y_min, self.y_max = min(self.y), max(self.y)

    # ==========================================
    # --- Internal Math Helpers (Batched) ---
    # ==========================================

    def _zero_cross_mask(self, field):
        """Finds 2x2 cells on a grid where the field crosses zero."""
        fs = np.sign(field).astype(np.int8)
        cell_max = np.maximum(fs[:-1, :-1], fs[:-1, 1:])
        np.maximum(cell_max, fs[1:, :-1], out=cell_max)
        np.maximum(cell_max, fs[1:, 1:], out=cell_max)
        
        cell_min = np.minimum(fs[:-1, :-1], fs[:-1, 1:])
        np.minimum(cell_min, fs[1:, :-1], out=cell_min)
        np.minimum(cell_min, fs[1:, 1:], out=cell_min)
        
        return (cell_max > 0) & (cell_min < 0)

    def _batched_plane_fit_2eq(self, candidate_coords, data1, data2):
        """
        Vectorized least-squares plane fit for 2 equations on N cells.
        Solves for the sub-pixel zero-crossing coordinates (dx, dy).
        """
        if len(candidate_coords) == 0: return np.array([]), np.array([]), np.array([], dtype=bool)
        
        y_idx, x_idx = candidate_coords[:, 0], candidate_coords[:, 1]
        
        # Extract the 4 corners of each candidate cell. Shape: (4, N)
        q1 = np.array([data1[y_idx, x_idx], data1[y_idx, x_idx+1], data1[y_idx+1, x_idx], data1[y_idx+1, x_idx+1]])
        q2 = np.array([data2[y_idx, x_idx], data2[y_idx, x_idx+1], data2[y_idx+1, x_idx], data2[y_idx+1, x_idx+1]])
        
        # Pseudoinverse for local cell coordinates [[0,0], [0,1], [1,0], [1,1]]
        A = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]], dtype=float)
        A_pinv = np.linalg.pinv(A)
        
        # Fit planes: p = [coeff_y, coeff_x, intercept]. Shape: (3, N)
        p1 = A_pinv @ q1  
        p2 = A_pinv @ q2
        
        # Setup Cramer's rule for A * [dy, dx]^T = b
        a11, a12 = p1[1], p1[0] # dx, dy for eq 1
        a21, a22 = p2[1], p2[0] # dx, dy for eq 2
        b1, b2 = -p1[2], -p2[2] # -intercepts
        
        det = a11*a22 - a12*a21
        valid = np.abs(det) > 1e-14
        
        dx = np.where(valid, ( a22*b1 - a12*b2) / det, -1)
        dy = np.where(valid, (-a21*b1 + a11*b2) / det, -1)
        
        return dx, dy, valid & (dx >= 0) & (dx < 1) & (dy >= 0) & (dy < 1)

    def _batched_plane_fit_3eq(self, candidate_coords, data1, data2, data3):
        """
        Vectorized least-squares plane fit solving normal equations for 3 equations on N cells.
        Used for overdetermined systems like L-points (codimension 2 but 3 components).
        """
        if len(candidate_coords) == 0: return np.array([]), np.array([]), np.array([], dtype=bool)
        
        y_idx, x_idx = candidate_coords[:, 0], candidate_coords[:, 1]
        
        q1 = np.array([data1[y_idx, x_idx], data1[y_idx, x_idx+1], data1[y_idx+1, x_idx], data1[y_idx+1, x_idx+1]])
        q2 = np.array([data2[y_idx, x_idx], data2[y_idx, x_idx+1], data2[y_idx+1, x_idx], data2[y_idx+1, x_idx+1]])
        q3 = np.array([data3[y_idx, x_idx], data3[y_idx, x_idx+1], data3[y_idx+1, x_idx], data3[y_idx+1, x_idx+1]])
        
        A_pinv = np.linalg.pinv(np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]], dtype=float))
        
        p1, p2, p3 = A_pinv @ q1, A_pinv @ q2, A_pinv @ q3
        
        # Design matrix elements across all N points
        A_y = np.array([p1[0], p2[0], p3[0]])  # Shape: (3, N)
        A_x = np.array([p1[1], p2[1], p3[1]])
        b   = np.array([-p1[2], -p2[2], -p3[2]])
        
        # Normal equations: A.T @ A * delta = A.T @ b
        AtA_00 = np.sum(A_y * A_y, axis=0)
        AtA_11 = np.sum(A_x * A_x, axis=0)
        AtA_01 = np.sum(A_y * A_x, axis=0)
        Atb_0 = np.sum(A_y * b, axis=0)
        Atb_1 = np.sum(A_x * b, axis=0)
        
        det = AtA_00*AtA_11 - AtA_01*AtA_01
        valid = np.abs(det) > 1e-14
        
        dy = np.where(valid, ( AtA_11*Atb_0 - AtA_01*Atb_1) / det, -1)
        dx = np.where(valid, (-AtA_01*Atb_0 + AtA_00*Atb_1) / det, -1)
        
        return dx, dy, valid & (dx >= 0) & (dx < 1) & (dy >= 0) & (dy < 1)

    def _batched_newton_raphson_2d(self, value_and_corrector, x0, y0, z_value, max_iter=10, tol=1e-6, value_tol=1e-6):
        """
        Batched Newton-Raphson evaluating all active points simultaneously via compute_cloud.
        Drops points as they converge to save computation.
        """
        x, y = x0.copy(), y0.copy()
        z = np.full_like(x, z_value)
        active = np.ones_like(x, dtype=bool)
        
        for _ in range(max_iter):
            if not np.any(active): break
            
            # Query the engine for the currently active subset of points
            fields = self.engine.compute_cloud(x[active], y[active], z[active], t=self.t, need_b=False)
            
            # Get values (M, 2) and Jacobian matrices (M, 2, 2) for active points
            v, C = value_and_corrector(fields.E, fields.jacobian_E) 
            
            # Exact 2x2 matrix inversion via Cramer's rule for batch (M,)
            det = C[:,0,0]*C[:,1,1] - C[:,0,1]*C[:,1,0]
            valid_det = np.abs(det) > 1e-14
            inv_det = np.where(valid_det, 1.0/det, 0.0)
            
            dx = ( C[:,1,1]*v[:,0] - C[:,0,1]*v[:,1]) * inv_det
            dy = (-C[:,1,0]*v[:,0] + C[:,0,0]*v[:,1]) * inv_det
            
            x[active] -= dx
            y[active] -= dy
            
            # Check convergence for this batch
            just_converged = (np.hypot(dx, dy) < tol) & valid_det
            active_indices = np.where(active)[0]
            
            # Turn off points that either converged or became singular
            active[active_indices[just_converged]] = False
            active[active_indices[~valid_det]] = False 
            
        # Final residual check across ALL points
        fields = self.engine.compute_cloud(x, y, z, t=self.t, need_b=False)
        final_v, _ = value_and_corrector(fields.E, fields.jacobian_E)
        return x, y, np.linalg.norm(final_v, axis=1) < value_tol

    def _find_singularities_template(self, z_value, candidate_coords, dx, dy, success, value_and_corrector_func, max_iter, tol, value_tol):
        """Standardizes the prediction -> correction -> bound-checking pipeline."""
        y_idx, x_idx = candidate_coords[success, 0], candidate_coords[success, 1]
        if len(x_idx) == 0: return np.array([]), np.array([]), np.array([]), np.array([]), np.array([], dtype=bool)
            
        cont_x = self.x[x_idx] + dx[success] * (self.x[x_idx+1] - self.x[x_idx])
        cont_y = self.y[y_idx] + dy[success] * (self.y[y_idx+1] - self.y[y_idx])
        
        final_x, final_y, confident = self._batched_newton_raphson_2d(
            value_and_corrector_func, cont_x, cont_y, z_value, max_iter, tol, value_tol
        )
        
        in_bounds = (final_x >= self.x_min) & (final_x <= self.x_max) & (final_y >= self.y_min) & (final_y <= self.y_max)
        return cont_x, cont_y, final_x, final_y, confident & in_bounds
    
    # ==========================================
    # --- 2D Point Finding Methods ---
    # ==========================================

    def _stokes_and_grads_from_EJ(self, E, J):
        """
        Calculates Stokes parameters and their spatial gradients.
        Works seamlessly for both single points and batched clouds (N,) arrays!
        """
        Ex, Ey = E[0], E[1]
        Ex_x, Ex_y, Ex_z = J[0]
        Ey_x, Ey_y, Ey_z = J[1]

        S0 = abs(Ex)**2 + abs(Ey)**2
        S1 = abs(Ex)**2 - abs(Ey)**2
        S2 = 2 * np.real(Ex * np.conj(Ey))
        S3 = 2 * np.imag(Ex * np.conj(Ey))

        S0_x = 2 * np.real(Ex_x * np.conj(Ex) + Ey_x * np.conj(Ey))
        S0_y = 2 * np.real(Ex_y * np.conj(Ex) + Ey_y * np.conj(Ey))
        S0_z = 2 * np.real(Ex_z * np.conj(Ex) + Ey_z * np.conj(Ey))

        S1_x = 2 * np.real(Ex_x * np.conj(Ex)) - 2 * np.real(Ey_x * np.conj(Ey))
        S1_y = 2 * np.real(Ex_y * np.conj(Ex)) - 2 * np.real(Ey_y * np.conj(Ey))
        S1_z = 2 * np.real(Ex_z * np.conj(Ex)) - 2 * np.real(Ey_z * np.conj(Ey))

        S2_x = 2 * np.real(Ex_x * np.conj(Ey) + Ex * np.conj(Ey_x))
        S2_y = 2 * np.real(Ex_y * np.conj(Ey) + Ex * np.conj(Ey_y))
        S2_z = 2 * np.real(Ex_z * np.conj(Ey) + Ex * np.conj(Ey_z))

        return ((S0, S1, S2, S3), np.array([S0_x, S0_y, S0_z]), np.array([S1_x, S1_y, S1_z]), np.array([S2_x, S2_y, S2_z]))
    
    def _stokes_c_point_value_and_corrector(self, E, J):
        """Returns normalized S1, S2 and their 2x2 Jacobian for the XY plane."""
        (S0, S1, S2, _), grad_S0, grad_S1, grad_S2 = self._stokes_and_grads_from_EJ(E, J)
        S0_safe = np.where(S0 > 1e-12, S0, 1.0)
        
        f_sp = np.stack([S1, S2], axis=-1) / S0_safe[:, None]
        J_sp_0 = (S0 * grad_S1[0] - S1 * grad_S0[0]) / S0_safe**2
        J_sp_1 = (S0 * grad_S1[1] - S1 * grad_S0[1]) / S0_safe**2
        J_sp_2 = (S0 * grad_S2[0] - S2 * grad_S0[0]) / S0_safe**2
        J_sp_3 = (S0 * grad_S2[1] - S2 * grad_S0[1]) / S0_safe**2
        
        # Build batched Jacobian: Shape (N, 2, 2)
        C = np.stack([np.stack([J_sp_0, J_sp_1], axis=-1), np.stack([J_sp_2, J_sp_3], axis=-1)], axis=1)
        return f_sp, C

    def find_stokes_C_points(self, z_value, E_grid, max_iter=10, pos_tol=1e-6, value_tol=1e-6):
        """
        Finds Stokes C-points, where transverse polarization (2D) is purely circular (s1=s2=0).

        Parameters
        ----------
        z_value : float
            z-coordinate of the observation plane.
        E_grid : np.ndarray
            Electric field array (3, ny, nx) evaluated at `z_value` from the engine.
        max_iter : int, optional
            Newton-Raphson iterations per candidate.
        pos_tol : float, optional
            Position convergence tolerance.
        value_tol : float, optional
            Residual tolerance for sqrt(s1^2 + s2^2).

        Returns
        -------
        list of dict
            List of dictionaries for each found singularity. Keys include:
            - 'position': (x, y, z) tuple of the refined root.
            - 'guess': (x, y, z) tuple of the initial plane-fit guess.
            - 'type': Morphological classification ('Star', 'Lemon', 'Monstar').
            - 'intensity': S0 value at the point.
            - 'handedness': Sign of S3 at the point (Right or Left circular).
            - 'confident': Boolean indicating if the solver fully converged.
        """
        Ex, Ey, _ = E_grid
        S0 = abs(Ex)**2 + abs(Ey)**2
        S0_safe = np.where(S0 == 0, 1.0, S0)
        s1, s2 = (abs(Ex)**2 - abs(Ey)**2) / S0_safe, (2 * np.real(Ex * np.conj(Ey))) / S0_safe

        mask = self._zero_cross_mask(s1) & self._zero_cross_mask(s2)
        coords = np.argwhere(mask)
        dx, dy, success = self._batched_plane_fit_2eq(coords, s1, s2)
        
        g_x, g_y, f_x, f_y, conf = self._find_singularities_template(
            z_value, coords, dx, dy, success, self._stokes_c_point_value_and_corrector, max_iter, pos_tol, value_tol
        )
        
        found_points = []
        if len(f_x) > 0:
            # Recompute topological properties for the refined points in one batch
            fields = self.engine.compute_cloud(f_x, f_y, np.full_like(f_x, z_value), t=self.t, need_b=False)
            all_S, _, S1_derivs, S2_derivs = self._stokes_and_grads_from_EJ(fields.E, fields.jacobian_E)
            S0_safe_cloud = np.where(all_S[0] > 1e-12, all_S[0], 1.0)
            
            S1_x, S1_y = S1_derivs[0]/S0_safe_cloud, S1_derivs[1]/S0_safe_cloud
            S2_x, S2_y = S2_derivs[0]/S0_safe_cloud, S2_derivs[1]/S0_safe_cloud
            D_I = S1_x * S2_y - S1_y * S2_x
            
            # Non-linear discriminant for Lemon vs Monstar
            NL_disc = ((2*S1_y + S2_x)**2 - 3*S2_y*(2*S1_x - S2_y)) * ((2*S1_x - S2_y)**2 + 3*S2_x*(2*S1_y + S2_x)) \
                      - (2*S1_x*S1_y + S1_x*S2_x - S1_y*S2_y + 4*S2_x*S2_y)**2

            for i in range(len(f_x)):
                c_type = 'Star' if D_I[i] < 0 else ('Lemon' if NL_disc[i] < 0 else 'Monstar')
                found_points.append({
                    'position': (f_x[i], f_y[i], z_value), 'guess': (g_x[i], g_y[i], z_value),
                    'type': c_type, 'intensity': float(all_S[0][i]), 'handedness': float(np.sign(all_S[3][i])),
                    'confident': bool(conf[i])
                })
        return found_points

    def _C_T_points_v_and_c(self, E, J):
        """Returns value and Jacobian for True 3D Circular polarization (E dot E = 0)."""
        E2 = np.sum(E*E, axis=0)
        dE2_dx, dE2_dy = 2 * np.sum(E * J[:, 0, :], axis=0), 2 * np.sum(E * J[:, 1, :], axis=0)
        
        f_cp = np.stack([np.real(E2), np.imag(E2)], axis=-1)
        C = np.stack([np.stack([np.real(dE2_dx), np.real(dE2_dy)], axis=-1),
                      np.stack([np.imag(dE2_dx), np.imag(dE2_dy)], axis=-1)], axis=1)
        return f_cp, C

    def find_C_T_points(self, z_value, E_grid, max_iter=10, pos_tol=1e-6, value_tol=1e-6):
        """
        Finds C^T points where true 3D circular polarization occurs (E·E = 0).

        Parameters
        ----------
        z_value : float
            z-coordinate of the observation plane.
        E_grid : np.ndarray
            Electric field array (3, ny, nx) evaluated at `z_value` from the engine.
        max_iter : int, optional
            Newton-Raphson iterations per candidate.
        pos_tol : float, optional
            Position convergence tolerance.
        value_tol : float, optional
            Residual tolerance for E·E.

        Returns
        -------
        list of dict
            List containing 'position', 'guess', and 'confident' status for each point.
        """
        E2 = np.sum(E_grid**2, axis=0)
        re_E2, im_E2 = np.real(E2), np.imag(E2)
        
        coords = np.argwhere(self._zero_cross_mask(re_E2) & self._zero_cross_mask(im_E2))
        dx, dy, success = self._batched_plane_fit_2eq(coords, re_E2, im_E2)
        
        g_x, g_y, f_x, f_y, conf = self._find_singularities_template(
            z_value, coords, dx, dy, success, self._C_T_points_v_and_c, max_iter, pos_tol, value_tol
        )
        return [{'position': (f_x[i], f_y[i], z_value), 'guess': (g_x[i], g_y[i], z_value), 'confident': bool(conf[i])} 
                for i in range(len(f_x))]

    def _L_T_points_v_and_c(self, E, J):
        """
        Computes the minimization step for Vector L-points where N = Re(E) x Im(E) = 0.
        Returns the Normal Equations (J.T @ J and J.T @ val) compatible with generic Newton solver.
        """
        ReE, ImE = np.real(E), np.imag(E)
        n_vec = np.cross(ReE, ImE, axis=0)
        
        dn_dx = np.cross(np.real(J[:, 0, :]), ImE, axis=0) + np.cross(ReE, np.imag(J[:, 0, :]), axis=0)
        dn_dy = np.cross(np.real(J[:, 1, :]), ImE, axis=0) + np.cross(ReE, np.imag(J[:, 1, :]), axis=0)
        
        C00, C11 = np.sum(dn_dx * dn_dx, axis=0), np.sum(dn_dy * dn_dy, axis=0)
        C01 = np.sum(dn_dx * dn_dy, axis=0)
        
        C = np.stack([np.stack([C00, C01], axis=-1), np.stack([C01, C11], axis=-1)], axis=1)
        v = np.stack([np.sum(dn_dx * n_vec, axis=0), np.sum(dn_dy * n_vec, axis=0)], axis=-1)
        return v, C

    def find_L_T_points(self, z_value, E_grid, max_iter=10, pos_tol=1e-6, value_tol=1e-6):
        """
        Finds L^T points where true 3D linear polarization occurs (Re(E) x Im(E) = 0).

        Parameters
        ----------
        z_value : float
            z-coordinate of the observation plane.
        E_grid : np.ndarray
            Electric field array (3, ny, nx) evaluated at `z_value` from the engine.
        max_iter : int, optional
            Newton-Raphson iterations per candidate.
        pos_tol : float, optional
            Position convergence tolerance.
        value_tol : float, optional
            Residual tolerance for the normal vector magnitude.

        Returns
        -------
        list of dict
            List containing 'position', 'guess', and 'confident' status for each point.
        """
        N_e = np.cross(np.real(E_grid), np.imag(E_grid), axis=0)
        coords = np.argwhere(
            self._zero_cross_mask(N_e[0]) & self._zero_cross_mask(N_e[1]) & self._zero_cross_mask(N_e[2])
            )
        
        dx, dy, success = self._batched_plane_fit_3eq(coords, N_e[0], N_e[1], N_e[2])
        g_x, g_y, f_x, f_y, conf = self._find_singularities_template(
            z_value, coords, dx, dy, success, self._L_T_points_v_and_c, max_iter, pos_tol, value_tol
        )
        return [{'position': (f_x[i], f_y[i], z_value), 'guess': (g_x[i], g_y[i], z_value), 'confident': bool(conf[i])} 
                for i in range(len(f_x))]

    # ==========================================
    # --- 3D Batched Line Tracing Methods ---
    # ==========================================

    def _batched_trace_lines(self, starting_points, val_jac_func, ds, max_steps, max_iter, value_tol):
        """
        Generic routine to trace N lines simultaneously in 3D.
        Uses a Predictor-Corrector method:
        1. Predictor: Steps along the curve's tangent vector (Gradient 1 x Gradient 2).
        2. Corrector: Refines the predicted point back onto the zero-curve via Minimum-Norm 
           pseudoinverse updates (underdetermined 2x3 Newton-Raphson).
        """
        pts = [p['position'] if isinstance(p, dict) else p for p in starting_points]
        if not pts: return []
        
        xyz = np.array(pts, dtype=float)
        N_lines = len(xyz)
        trajectories = [[xyz[i].copy()] for i in range(N_lines)]
        
        # Track which lines are still successfully being traced
        active = np.ones(N_lines, dtype=bool)
        
        for _ in range(max_steps):
            if not np.any(active): break
            
            # --- 1. Predictor step (All Active Lines) ---
            fields = self.engine.compute_cloud(xyz[active, 0], xyz[active, 1], xyz[active, 2], t=self.t, need_b=False)
            vals, J = val_jac_func(fields.E, fields.jacobian_E) # J is (M, 2, 3) where M is num active lines
            
            # Tangent vector is cross product of the gradients of the two conditions
            tangent = np.cross(J[:, 0, :], J[:, 1, :])
            norm_t = np.linalg.norm(tangent, axis=1)
            valid_t = norm_t > 1e-12
            tangent[valid_t] = tangent[valid_t] / norm_t[valid_t, None]
            
            pred_xyz = np.zeros_like(xyz[active])
            pred_xyz[valid_t] = xyz[active][valid_t] + tangent[valid_t] * ds
            
            active_indices = np.where(active)[0]
            active[active_indices[~valid_t]] = False # Kill flat tangent lines (likely topological anomalies or bounds)
            
            # --- 2. Corrector step (Sub-loop to relax onto line) ---
            corr_xyz = pred_xyz[valid_t]
            corr_active = np.ones(len(corr_xyz), dtype=bool)
            corr_success = np.zeros(len(corr_xyz), dtype=bool)
            
            for _ in range(max_iter):
                if not np.any(corr_active): break
                c_fields = self.engine.compute_cloud(corr_xyz[corr_active, 0], corr_xyz[corr_active, 1], corr_xyz[corr_active, 2], t=self.t, need_b=False)
                c_v, c_J = val_jac_func(c_fields.E, c_fields.jacobian_E)
                
                # Check convergence
                c_converged = np.linalg.norm(c_v, axis=1) < value_tol
                just_converged = c_converged & corr_active
                corr_success[just_converged] = True
                corr_active[just_converged] = False
                
                not_conv = ~c_converged
                if np.any(not_conv):
                    # Minimum Norm Pseudoinverse: Delta = J.T * (J * J.T)^-1 * vals
                    nc_J, nc_v = c_J[not_conv], c_v[not_conv]
                    JJT_00, JJT_11 = np.sum(nc_J[:,0,:]**2, axis=1), np.sum(nc_J[:,1,:]**2, axis=1)
                    JJT_01 = np.sum(nc_J[:,0,:] * nc_J[:,1,:], axis=1)
                    
                    det = JJT_00*JJT_11 - JJT_01**2
                    vdet = np.abs(det) > 1e-14
                    inv_det = np.where(vdet, 1.0/det, 0.0)
                    
                    # Solve (J * J.T) * lambda = vals
                    lam_0 = ( JJT_11*nc_v[:,0] - JJT_01*nc_v[:,1]) * inv_det
                    lam_1 = (-JJT_01*nc_v[:,0] + JJT_00*nc_v[:,1]) * inv_det
                    
                    # Update X = X - J.T * lambda
                    update_idx = np.where(corr_active)[0][not_conv]
                    corr_xyz[update_idx, 0] -= nc_J[:,0,0]*lam_0 + nc_J[:,1,0]*lam_1
                    corr_xyz[update_idx, 1] -= nc_J[:,0,1]*lam_0 + nc_J[:,1,1]*lam_1
                    corr_xyz[update_idx, 2] -= nc_J[:,0,2]*lam_0 + nc_J[:,1,2]*lam_1
                    
                    corr_active[update_idx[~vdet]] = False # Kill singular lines
            
            # --- 3. Apply successful steps to global trajectory tracker ---
            valid_active_indices = active_indices[valid_t]
            active[valid_active_indices[~corr_success]] = False # Kill globally if Corrector failed
            
            success_local = np.where(corr_success)[0]
            success_global = valid_active_indices[success_local]
            xyz[success_global] = corr_xyz[success_local]
            
            for local_i, global_i in zip(success_local, success_global):
                trajectories[global_i].append(corr_xyz[local_i].copy())

        return [np.array(t) for t in trajectories]

    def _stokes_C_val_jac_3d(self, E, J):
        """Returns normalized [S1, S2] and full 2x3 Jacobian for tracing Stokes C-lines."""
        (S0, S1, S2, _), grad_S0, grad_S1, grad_S2 = self._stokes_and_grads_from_EJ(E, J)
        S0_safe = np.where(S0 > 1e-12, S0, 1.0)
        vals = np.stack([S1, S2], axis=-1) / S0_safe[:, None]
        jac = np.stack([(S0 * grad_S1 - S1 * grad_S0).T / S0_safe**2, 
                        (S0 * grad_S2 - S2 * grad_S0).T / S0_safe**2], axis=1)
        return vals, jac

    def trace_stokes_C_lines(self, starting_points, ds=0.05, max_steps=500, max_iter=10, value_tol=1e-6):
        """
        Traces Stokes C-lines in 3D (curves where s1 = s2 = 0) originating from seed points.

        Parameters
        ----------
        starting_points : list of dict or list of tuple
            Seed points. Can be the dictionaries returned by `find_stokes_C_points()` 
            or raw (x,y,z) coordinate tuples.
        ds : float, optional
            Step size spatial parameter for tracing along the tangent.
        max_steps : int, optional
            Maximum number of tracing steps per line before giving up/stopping.
        max_iter : int, optional
            Corrector (Newton) iterations allowed per spatial step.
        value_tol : float, optional
            Residual tolerance for the line condition (sqrt(s1^2 + s2^2)).

        Returns
        -------
        list of np.ndarray
            List containing a trajectory array of shape (N_steps, 3) for each seed point.
        """
        return self._batched_trace_lines(starting_points, self._stokes_C_val_jac_3d, ds, max_steps, max_iter, value_tol)

    def _C_T_val_jac_3d(self, E, J):
        """Returns [Re(E^2), Im(E^2)] and 2x3 Jacobian for Vector C^T lines."""
        E2 = np.sum(E*E, axis=0)
        dE2_dx, dE2_dy, dE2_dz = 2 * np.sum(E * J[:, 0, :], axis=0), 2 * np.sum(E * J[:, 1, :], axis=0), 2 * np.sum(E * J[:, 2, :], axis=0)
        return np.stack([np.real(E2), np.imag(E2)], axis=-1), \
               np.stack([np.stack([np.real(dE2_dx), np.real(dE2_dy), np.real(dE2_dz)], axis=-1),
                         np.stack([np.imag(dE2_dx), np.imag(dE2_dy), np.imag(dE2_dz)], axis=-1)], axis=1)

    def trace_C_T_lines(self, starting_points, ds=0.05, max_steps=500, max_iter=10, value_tol=1e-6):
        """
        Traces C^T lines in 3D (curves where E·E = 0) originating from seed points.

        Parameters
        ----------
        starting_points : list of dict or list of tuple
            Seed points. Can be dictionaries from `find_C_T_points()` or (x,y,z) tuples.
        ds : float, optional
            Step size spatial parameter for tracing along the tangent.
        max_steps : int, optional
            Maximum number of tracing steps per line.
        max_iter : int, optional
            Corrector (Newton) iterations allowed per spatial step.
        value_tol : float, optional
            Residual tolerance for the line condition (abs(E·E)).

        Returns
        -------
        list of np.ndarray
            List containing a trajectory array of shape (N_steps, 3) for each seed point.
        """
        return self._batched_trace_lines(starting_points, self._C_T_val_jac_3d, ds, max_steps, max_iter, value_tol)

    def _L_T_val_jac_3d(self, E, J):
        """
        Returns values and 2x3 Jacobian for Vector L-Lines (True Linear Polarization).
        
        Math Note:
        The condition N = Re(E) x Im(E) = 0 is codimension 2.
        The components of N are dependent (N is orthogonal to E). Therefore, vanishing 
        of any two components implies vanishing of the third.
        We solve for the intersection of N_y = 0 and N_z = 0 to trace the curve.
        """
        ReE, ImE = np.real(E), np.imag(E)
        n_vec = np.cross(ReE, ImE, axis=0)
        
        dn_dx = np.cross(np.real(J[:, 0, :]), ImE, axis=0) + np.cross(ReE, np.imag(J[:, 0, :]), axis=0)
        dn_dy = np.cross(np.real(J[:, 1, :]), ImE, axis=0) + np.cross(ReE, np.imag(J[:, 1, :]), axis=0)
        dn_dz = np.cross(np.real(J[:, 2, :]), ImE, axis=0) + np.cross(ReE, np.imag(J[:, 2, :]), axis=0)
        
        return np.stack([n_vec[1], n_vec[2]], axis=-1), \
               np.stack([np.stack([dn_dx[1], dn_dy[1], dn_dz[1]], axis=-1),
                         np.stack([dn_dx[2], dn_dy[2], dn_dz[2]], axis=-1)], axis=1)

    def trace_L_lines(self, starting_points, ds=0.05, max_steps=500, max_iter=10, value_tol=1e-6):
        """
        Traces L^T lines in 3D (curves where Re(E) x Im(E) = 0) originating from seed points.

        Parameters
        ----------
        starting_points : list of dict or list of tuple
            Seed points. Can be dictionaries from `find_L_T_points()` or (x,y,z) tuples.
        ds : float, optional
            Step size spatial parameter for tracing along the tangent.
        max_steps : int, optional
            Maximum number of tracing steps per line.
        max_iter : int, optional
            Corrector (Newton) iterations allowed per spatial step.
        value_tol : float, optional
            Residual tolerance for the line condition (abs(N_y) and abs(N_z)).

        Returns
        -------
        list of np.ndarray
            List containing a trajectory array of shape (N_steps, 3) for each seed point.
        """
        return self._batched_trace_lines(starting_points, self._L_T_val_jac_3d, ds, max_steps, max_iter, value_tol)