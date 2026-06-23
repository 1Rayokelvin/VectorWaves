This page details the core methods, conventions, and topological concepts utilized in VectorWaves.

## 1. Discrete Plane-Wave Expansions

VectorWaves constructs fields through the superposition of plane waves. To sample the angular spectrum efficiently and without bias, `BeamMaker` samples wavevectors over the source's solid angle using a **Fibonacci-sphere quadrature**.

This approach provides near-uniform angular density across the sphere, avoiding the polar clustering issues associated with a standard latitude/longitude grid. The accuracy of any computed field quantity scales with `config.source.num_modes`. For typical simulations, a few thousand modes are sufficient with proper tuning of source's solid angle i.e. `config.source.theta_max`.

### Power Normalization and Intensity Scale
Modes are constructed such that `sum(mode_irradiances) = 1` for a unit-amplitude source. The `config.source.intensity_scale` enters as `sqrt(intensity_scale)` on the amplitude, meaning total power scales linearly. 

For polychromatic fields, spectral profile weights are L2-normalized before scaling, ensuring total power is preserved across different spectral shapes. For amplitude randomization (`randomize.amplitude`), each mode's amplitude is multiplied by a unit-variance complex Gaussian draw scaled by `1/sqrt(2)`. This preserves expected total power while successfully introducing speckle statistics.

*(Note: `intensity_E` is a real-space quantity representing the squared field magnitude evaluated point by point, carrying the power normalization set during beam construction.)*

## 2. Polarization Transport

To ensure that the polarization of each plane-wave mode is physically consistent with the overall beam, VectorWaves utilizes **Rodrigues rotation** for polarization transport.

The user defines a base Jones vector, `config.source.pol_vect = (px, py)`, relative to the macroscopic propagation axis `config.source.beam_axis`. `BeamMaker` then carries this vector to each individual plane-wave mode via a Rodrigues rotation that maps the beam axis unit vector to the specific mode's wavevector `k_hat`. This is done for a geometrically determined polarization state.

## 3. Conventions & Units

- **Units and Phase:** VectorWaves uses natural units where `c = 1`. The time-harmonic phase accumulated by each plane-wave mode is `phase = kx·x + ky·y + kz·z − ω·t`. Therefore, time `t` is defined in units of length (equivalent to `ct` in SI). Wavenumber `k` is stored in `rad / spatial_unit`, where the spatial unit is determined by `op.size` and `op.spacing`.
- **Beam Axis:** The default propagation axis is `z`. `compute_on_op` returns field values on the transverse (xy) plane at a given `z`, while `compute_cloud` and `compute_point` evaluate arbitrary 3D spatial points directly.
- **Jacobian Indexing:** Spatial derivatives follow the convention `jacobian_E[i, j, ...]` representing `dE_i / dx_j` (first index is field component, second is derivative direction). `div_E` is its trace; `curl_E` arises from its antisymmetric part.
- **Polarization Handedness:** `+1` indicates Left Circular Polarization (LCP) / Counter-Clockwise (CCW) rotation. `−1` indicates Right Circular Polarization (RCP) / Clockwise (CW) rotation. This convention is determined by the sign of the Stokes parameter `S3` and is used consistently throughout the singularity finders.

## 4. Polarization Topology & Singularities

In non-paraxial fields, polarization ellipses are not confined to the transverse plane. The ellipse lives in a plane defined by its major and minor axes, with an area vector given by `Re(E) × Im(E)`. Two natural, degenerate topological singularities arise in full 3D:

- **Lᵀ-points (True 3D Linear Polarization):** Occur where `Re(E) × Im(E) = 0`. The area vector vanishes, meaning the polarization ellipse collapses strictly to a line. Found by `find_C_T_points`.
- **Cᵀ-points (True 3D Circular Polarization):** Occur where `E·E = 0` (complex dot product, no conjugate). In the plane of the ellipse, the major and minor axes are perfectly equal. Found by `find_L_T_points`.

### Stokes Projections
The Stokes C-points found by `find_stokes_C_points`, where `s1 = s2 = 0`, equivalently `Ex² + Ey² = 0` (ignoring Ez, transverse projection). In the paraxial limit, Ez is negligible, making Stokes C-points and Cᵀ-points nearly identical. However, for strongly non-paraxial fields (e.g., tight focuses or isotropic sources), Ez is significant. Here, the two singularity sets become geometrically distinct, and transverse Stokes C-points become biased approximations.

### Singularity Refinement
To precisely locate these singularities, finders first locate candidates on the discrete `E_grid`, then refine their positions to sub-pixel accuracy against a specific residual:

| Singularity | Condition | Solver | Residual |
|---|---|---|---|
| **Stokes C-point** | `s1 = s2 = 0` | Newton-Raphson | `sqrt(s1² + s2²)` |
| **Cᵀ-point** | `E·E = 0` | Newton-Raphson | `\|E·E\|` |
| **Lᵀ-point** | `Re(E) × Im(E) = 0` | Gauss-Newton | `\|Re(E) × Im(E)\|` |

While Stokes C-points and Cᵀ-points are exact square systems (2 conditions, 2 unknowns in a plane) solvable by Newton-Raphson, Lᵀ-points appear overdetermined (3 scalar conditions on 2 unknowns). Because the area vector `N` is always orthogonal to `E`, its components aren't independent. The solver utilizes Gauss-Newton to minimize `|N|²` instead.

### A Note on Tolerances
The Stokes C-point residual, `sqrt(s1² + s2²)`, is dimensionless since the normalized Stokes parameters absorb the local intensity `S0`. The default `value_tol = 1e-6` works reliably regardless of source power. 

Conversely, the residuals for Cᵀ and Lᵀ points possess units of intensity. In regions of low intensity—such as speckle dark spots or nodal lines—these absolute residuals can artificially fall below the threshold or induce instability. Tuning `config.source.intensity_scale` helps bring the field into a numerical regime where these dimensional tolerances are meaningful. 

### Line Tracing
Singularities form structures in 3D. `trace_stokes_C_lines`, `trace_C_T_lines`, and `trace_L_lines` walk outward from refined seed points along the local tangent in steps of size `ds`, utilizing a Newton corrector to continuously re-project each step back onto the singular condition manifold.
