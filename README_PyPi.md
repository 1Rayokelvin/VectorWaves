VectorWaves is a Python library for constructing and analyzing electromagnetic fields through discrete plane-wave expansions.

It provides tools for generating structured optical fields, evaluating fully three-dimensional electric and magnetic fields, and analyzing polarization singularities such as C-points, Cᵀ-points, and Lᵀ-points.

## Installation

```
pip install vectorwaves
```

## Quick Example

```
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
![LG beam intensity](https://github.com/1Rayokelvin/VectorWaves/blob/main/docs/images/LG_beam.png)

## Features

- Structured optical field generation
- Monochromatic and polychromatic sources
- Three-dimensional electromagnetic field evaluation
- GPU acceleration through CuPy
- Polarization singularity analysis

## Documentation

For tutorials, examples, API documentation, and the underlying physics, see:

* GitHub: https://github.com/1RayOfKelvin/VectorWaves
* Documentation: https://1rayokelvin.github.io/VectorWaves