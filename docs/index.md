# VectorWaves Documentation

VectorWaves is a Python library for constructing and analyzing electromagnetic fields through discrete plane-wave expansions.

If you're looking for a quick introduction with simple examples, see the [README](https://github.com/1rayokelvin/VectorWaves).

## Tutorials

- [Scalar fields](tutorials/Scalar_tutorial.ipynb): verifying known scalar-field phenomenology, where non-paraxiality breaks the scalar picture, and the Wolf-type vectorial effects that emerge from it. Also covers custom k-space spectra and polychromatic fields.
- [Vector fields](tutorials/Vector_tutorial.ipynb): constructing E, B, and the Poynting vector directly, decomposing into the left/right circular basis to see C-points as phase singularities, and scaling laws for C-point density in speckle fields.

## Physics

- [Conventions and methods](physics.md) — unit/phase/coordinate conventions, plus the reasoning behind Fibonacci-sphere sampling, parallel transport, and singularity refinement.

## API Reference

- [Config](api/config.md)
- [Core Objects](api/core.md)
- [Singularities](api/singularities.md)
- [Utilities](api/utils.md)