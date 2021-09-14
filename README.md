# NBodySim

This project is a GPU-powered N-body gravity simulation for Python.

Powered by [taichi](https://github.com/taichi-dev/taichi), this code solves the Poisson equation to determine the forces on the millions of particles.

Project next steps:

- [ ] Use adaptive mesh refinement to accelerate solving of Poisson's equation.
- [ ] Use Newton's law of gravity for all sufficiently close particles.
- [ ] Expand the simulation to 3D and rasterize into 2D space.
