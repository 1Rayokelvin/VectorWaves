VectorWaves provides a framework for generating, computing, and analyzing fully three-dimensional electromagnetic fields and their topology through discrete plane-wave expansions.

## Installation

```bash
pip install vectorwaves
```

For additional features, you can install the optional dependencies:

| Extra | Purpose |
|---------|---------|
| `viz` | Matplotlib and PyVista for visualizations |
| `progress` | Progress bars via tqdm |
| `gpu` | CUDA acceleration via CuPy |
| `all` | All the above |

To install, 
```bash
pip install vectorwaves[extra]
```

## Features

- Physics-oriented configuration system with `numpy`, `numba`, and CuPy (GPU) backends.
- Exact non-paraxial 3D propagation via Fibonacci-sphere discrete plane-wave expansions.
- Monochromatic and polychromatic sources with arbitrary envelopes, structured light support.
- Stochastic process generation for speckle-like fields.
- Fully analytic computation of E-fields, B-fields, spatial derivatives.
- Topological polarization analysis: C, Cᵀ, and Lᵀ point finding with 3D line tracing.
  
## Quick Example

```python
import vectorwaves as vw

# Generate a tightly-focused Laguerre-Gaussian beam
config = vw.get_config()
config.source.k_space.laguerre_gauss(p=1, l=2, sigma_k_perp=1)
config.source.randomize.off()

# Construct the beam and visualize its plane-wave modes
beam = vw.setup_beam(config)

# Requires matplotlib, install with  'viz' extra: pip install vectorwaves[viz]
beam.plot_kspace_3d(plot_type='colored_vectors')
```

![LG beam k-space](https://github.com/1Rayokelvin/VectorWaves/blob/main/docs/images/LG_kspace.png?raw=true)

For tutorials and examples, please refer to the [official documentation](https://1rayokelvin.github.io/VectorWaves). Source code is available on [GitHub](https://github.com/1Rayokelvin/VectorWaves/).
