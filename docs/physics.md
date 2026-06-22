## Conventions

**Units and phase.** Natural units, `c = 1`. The time-harmonic phase accumulated by each plane-wave mode is

```
phase = kx·x + ky·y + kz·z − ω·t
```

so `t` is in units of length (= `ct` in SI). `k` is stored in `rad / spatial_unit`, where spatial unit is whatever you set `op.size` and `op.spacing` in.

**Beam axis and transverse plane.** Default beam propagation axis is `z`. `compute_on_op` returns field values on the transverse plane at a given `z`; `compute_cloud` and `compute_point` evaluate arbitrary non-transverse points directly.

**Jacobian indexing.** `jacobian_E[i, j, ...]` is `dE_i / dx_j` — first index is the field component (Ex, Ey, Ez), second is the derivative direction (x, y, z). `div_E` is its trace; `curl_E` comes from its antisymmetric part.

**Polarization basis.** The base Jones vector (`config.source.pol_vect = (px, py)`) is defined relative to the beam axis. Both basis vectors are normalized on init. Handedness convention: `+1` = LCP / CCW, `−1` = RCP / CW, determined by the sign of `S3`. Used consistently in `get_pol_ellipse_params` and in the `handedness` field returned by all singularity finders.

**Power normalization and intensity scale.** Modes are constructed so that `sum(mode_irradiances) = 1` for a unit-amplitude source. `config.source.intensity_scale` enters as `sqrt(intensity_scale)` on the amplitude, so power scales linearly. For polychromatic fields, the spectral profile weights are additionally L2-normalized before scaling, so total power is preserved across different spectral shapes.

For amplitude randomization (`randomize.amplitude`), each mode's amplitude is multiplied by a unit-variance complex Gaussian draw scaled by `1/sqrt(2)`, preserving the expected total power while adding speckle statistics.

`intensity_E` is real-space: it is the squared field magnitude evaluated point by point on the observation plane, carrying the power normalization set during beam construction. It is not a k-space quantity.

---

## Methods

**Fibonacci-sphere sampling.** `BeamMaker` samples plane-wave wavevectors over the source's solid angle using a Fibonacci-sphere quadrature, which gives near-uniform angular density across the sphere without the polar clustering of a latitude/longitude grid. The accuracy of any field quantity scales with `config.source.num_modes`; for smooth structured beams a few thousand modes are typically sufficient, while speckle or highly non-paraxial sources require more.

**Polarization transport (Rodrigues rotation).** The base Jones vector `(px, py)` is defined on the beam axis. `BeamMaker` carries it to each plane-wave mode via a Rodrigues rotation that takes the beam axis unit vector to the mode's `k_hat`. This means every mode in the angular spectrum inherits a geometrically consistent polarization state, and no per-mode polarization specification is needed. Per-mode polarization vectors can be recovered after construction as `c / a` (vector amplitude divided by scalar amplitude).

**Singularity conditions and their physical meaning.**

In a non-paraxial field, the polarization ellipse is not confined to the transverse plane — it lives in a plane defined by its major and minor axes, and the ellipse's area vector is `Re(E) × Im(E)`. Two natural degenerate conditions arise:

- **Lᵀ-points** (`Re(E) × Im(E) = 0`): the area vector vanishes, meaning the polarization ellipse has collapsed to a line — *true* 3D linear polarization. The area vector being zero is the exact statement that no ellipse exists.

- **Cᵀ-points** (`E·E = 0`): in the plane of the polarization ellipse, the major and minor axes are equal — *true* 3D circular polarization. `E·E = Ex²+Ey²+Ez²` (complex dot product, no conjugate), so this is a complex condition with real and imaginary parts both vanishing.

The Stokes C-points found by `find_stokes_C_points` are the *transverse projection* of this: `s1 = s2 = 0` sets only `Ex²+Ey² = 0` (ignoring Ez). In the paraxial limit Ez ≈ 0 and Stokes C-points and Cᵀ-points nearly coincide. For strongly non-paraxial fields (tight focus, maximally divergent beam, isotropic source), Ez is no longer negligible and the two singularity sets are geometrically distinct — `find_stokes_C_points` is a biased approximation of circular polarization in that regime.

The analogous story for L is different in codimension. Stokes L lines are codimension-2 objects (surfaces in 3D, not lines), whereas Lᵀ lines are codimension-2 *lines* — so unlike the C/Cᵀ case, a projection of a line is always a line: every Lᵀ point projects onto an L point, but not conversely.

**Singularity refinement.** All finders follow the same two-stage pattern: locate candidates on the discrete `E_grid`, then refine to sub-pixel accuracy against a type-specific residual:

| Singularity | Condition | Solver | Residual |
|---|---|---|---|
| Stokes C-point | `s1 = s2 = 0` (circular, transverse) | Newton-Raphson | `sqrt(s1² + s2²)` |
| Cᵀ-point | `E·E = 0` (circular, full 3D) | Newton-Raphson | `\|E·E\|` |
| Lᵀ-point | `Re(E) × Im(E) = 0` (linear, full 3D) | Gauss-Newton | `\|Re(E) × Im(E)\|` |

Stokes C-points and Cᵀ-points are square systems (2 real conditions, 2 unknowns in the plane), so standard Newton-Raphson applies directly.

Lᵀ-points are different. The condition `N = Re(E) × Im(E) = 0` naively looks like 3 scalar conditions on 2 unknowns — overdetermined. However, since `N` is always orthogonal to `E`, the three components of `N` are not independent, and the true codimension is 2. The solver handles this by forming the normal equations (`Jᵀ J` and `Jᵀ N`) and solving with Gauss-Newton, which minimizes `|N|²` over the plane rather than requiring an exact square system.

Default tolerances are `pos_tol = value_tol = 1e-6` with `max_iter = 10`. A point is marked `confident = True` if both converge within budget; non-confident points should be filtered before use (as in the README example).

**A note on `value_tol` and intensity scale.** The three residuals have different units:

- Stokes C-point: `sqrt(s1² + s2²)` is dimensionless (normalized Stokes parameters absorb the local intensity). `value_tol = 1e-6` is an absolute fraction, scale-independent, and will work correctly regardless of source power.
- Cᵀ-point: `|E·E|` has units of intensity. `value_tol = 1e-6` is only meaningful relative to the local field amplitude.
- Lᵀ-point: `|Re(E) × Im(E)|` also has units of intensity.

For sources with unusual power spectra or low total power, `|E·E|` and `|Re(E) × Im(E)|` can sit below `1e-6` everywhere — making the condition trivially "met" near every candidate — or near machine precision, causing 1/small-number instability in the Jacobian. In both cases, tuning `config.source.intensity_scale` (which enters as `sqrt(intensity_scale)` on all mode amplitudes) brings the field up to a regime where the dimensional tolerances are meaningful.

This is also why regions of genuinely low intensity — speckle dark spots, beam edges, nodal lines — are unreliable for Cᵀ and Lᵀ refinement even at normal power levels. The Stokes C-point finder is immune to this by construction, since normalization by `S0` (local intensity) is built into the Stokes parameters themselves, though that same division by `S0` will become numerically unstable if `S0` is near zero, so dark-spot candidates should be treated with scepticism regardless of `confident` status.

Stokes C-points are additionally classified morphologically as `'Star'`, `'Lemon'`, or `'Monstar'` from the local index structure.

**Line tracing.** `trace_stokes_C_lines` / `trace_C_T_lines` / `trace_L_lines` walk outward from refined seed points in steps of size `ds` (default 0.05) along the local tangent, with a Newton corrector re-projecting each step back onto the singular condition. Output is one `(N_steps, 3)` trajectory array per seed; maximum steps per line is `max_steps` (default 500).

For Lᵀ line tracing specifically, the same `N ⊥ E` dependence is used differently: rather than minimizing all three components, the tracer solves only `Ny = 0` and `Nz = 0` as two independent scalar conditions (vanishing of any two implies vanishing of the third), giving a well-posed 2×3 system for the tangent step that standard Newton handles directly.

**Time dependence.** Singularity finding defaults to `t = 0`. For monochromatic fields this is always correct — polarization topology is time-independent. For polychromatic fields, where the field envelope evolves in time, set `finder.t` explicitly to evaluate singularities at a specific moment.