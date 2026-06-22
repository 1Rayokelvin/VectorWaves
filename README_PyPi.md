VectorWaves provides a framework for generating, computing, and analyzing fully three-dimensional electromagnetic fields through discrete plane-wave expansions.

## Installation
```
pip install vectorwaves
```

## Features

- Physics-oriented hierarchical configuration system
- Monochromatic sources including Gaussian and Laguerre-Gaussian beams, with support for custom source definitions
- Polychromatic sources with Gaussian, Lorentzian, and custom spectral distributions
- Polarization singularity analysis (C, Cᵀ, and Lᵀ points, including their 3D counterparts)
- GPU acceleration through CuPy

## Quick Example

```python
import matplotlib.pyplot as plt
import vectorwaves as vw

# specifying the system
config = vw.get_config()
config.source.k_space.laguerre_gauss(p=1, l=2, sigma_k_perp=0.5)
config.source.randomize.off()

# computing fields
engine = vw.setup_engine(config)
result = engine.compute_on_op(z=0.0)

# plotting profile
plt.imshow(result.intensity_E, cmap="magma")
plt.title("LG beam Intensity profile")
plt.show()
```
Output

![LG beam Intensity](https://github.com/1Rayokelvin/VectorWaves/blob/main/docs/images/LG_beam.png?raw=true)

For tutorials and examples, please refer to [documentation](https://1rayokelvin.github.io/VectorWaves). Source code is available on [GitHub](https://github.com/1Rayokelvin/VectorWaves/).
