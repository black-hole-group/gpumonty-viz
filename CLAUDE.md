# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPUMonty Visualization Suite — a collection of Python scripts for visualizing photon geodesic trajectories in Kerr spacetime around black holes, with optional GRMHD density field overlays. This is a post-processing toolkit for output from GPUmonty (`../gpumonty/`), a CUDA-based Monte Carlo radiative transfer code.

## Running Scripts

No build system. All scripts are run directly:

```bash
# Quick interactive 3D visualization (Plotly, no volume rendering)
python src/plot_geodesics.py output/geodesics.h5 --n 100

# Static image with volume-rendered density (yt backend)
python src/plot_geodesics_yt.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --n 50 --output image.png

# Movie: geodesics progressively build up (yt + ffmpeg)
python src/movie_geodesics_yt.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --n 50 --n-frames 100 --fps 30

# Parallel movie rendering (distribute frames across CPU cores)
python src/movie_parallel.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --n 50 --workers 8

# Movie where camera follows a specific photon (PyVista backend)
python src/movie_follow.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --follow 0 --n 50 --output follow_movie.mp4

# Interactive camera exploration
jupyter notebook src/interactive_camera_pv.ipynb
```

**Common CLI flags** (most scripts):
- `--n N`: Number of geodesics to display (randomly sampled)
- `--r-max R`: Clipping radius in gravitational radii
- `--resolution N`: Cartesian interpolation grid size (256³ typical)
- `--cam-position X Y Z`: Camera placement in r_g units
- `--fps N`: Video framerate
- `--output PATH`: Output file path

## Dependencies

```
h5py, numpy, scipy, tqdm         # Always required
yt                                # For *_yt.py scripts
pyvista                           # For *_pv.py scripts and movie_follow.py
plotly                            # For plot_geodesics.py
pyvista[jupyter], trame           # For interactive_camera_pv.ipynb
ffmpeg                            # External, for video assembly
```

## Architecture

### Data Flow

```
GPUmonty (../gpumonty/) CUDA simulation
    → geodesics.h5       (photon trajectories in Boyer-Lindquist coords)
    → dump_SANE.h5       (GRMHD primitives on Modified Kerr-Schild grid)
        ↓
Python scripts:
    1. Load geodesics (HDF5)
    2. Load GRMHD density (HDF5)
    3. Coordinate transforms: MKS → Boyer-Lindquist → Cartesian
    4. Interpolate GRMHD data onto uniform Cartesian grid (scipy.ndimage)
    5. Render scene
        ↓
    PNG / MP4 / interactive HTML
```

### Two Rendering Backends

**yt** (`plot_geodesics_yt.py`, `movie_geodesics_yt.py`, `movie_parallel.py`): Volume-renders the GRMHD density field with geodesic line overlays. Better for static images and straightforward frame sequences.

**PyVista/VTK** (`plot_geodesics_pv.py`, `movie_follow.py`, `interactive_camera_pv.ipynb`): Advanced camera control (e.g., camera riding along a photon path). Supports off-screen rendering for batch frame generation. `plot_geodesics_pv.py` is a helper module imported by the others.

**Plotly** (`plot_geodesics.py`): Lightweight, no volume rendering. Best for quick interactive inspection of geodesic geometry.

### Coordinate Systems

- **Modified Kerr-Schild (MKS)**: Internal GRMHD grid coordinates `(x1, x2, x3)`; `x2` is a nonlinear function of θ controlled by `hslope`. Requires lookup-table inversion to recover θ.
- **Boyer-Lindquist (BL)**: `(r, θ, φ)` — used directly in `geodesics.h5` output from GPUmonty
- **Cartesian**: Final visualization space, converted from BL via standard Kerr formulas

The MKS→BL→Cartesian transformation is the core shared logic replicated across all scripts.

### HDF5 Input Formats (from `../gpumonty/src/`)

**`geodesics.h5`** — written by `save_geodesics_h5()` in `../gpumonty/src/main.cu`:
```
r          [nph, max_saved_steps]   Boyer-Lindquist radial coordinate
theta      [nph, max_saved_steps]   Boyer-Lindquist polar angle
phi        [nph, max_saved_steps]   Azimuthal angle
nsteps     [nph]                    Valid steps per photon (rest are zeros)
nph                                 Total photons traced
max_saved_steps                     trace_maxsteps / trace_stride + 1
trace_stride                        Step interval at which positions were saved
trace_maxsteps                      Max integration steps per photon
```
Photons hitting the horizon or escaping terminate early; `nsteps[i]` gives the valid slice length for photon `i`.

**`dump_SANE.h5`** (iharm3d format) — read by `../gpumonty/src/iharm_model/model.cu`:
```
/header/n1, n2, n3                  Grid dimensions (radial, poloidal, azimuthal)
/header/geom/startx1..3             Coordinate lower bounds (MKS)
/header/geom/dx1..3                 Coordinate spacing (MKS)
/header/geom/mks/a                  Black hole spin parameter
/header/geom/mks/hslope             Latitude grid refinement (θ compression)
/header/geom/mks/r_in, r_out        Boyer-Lindquist inner/outer radius
/prims/                             [N1, N2, N3, N_PRIM] primitive variables
                                    Index 0=density (KRHO) — used for visualization
/t                                  Simulation time
```

**Generating geodesics**: Add to the GPUmonty `.par` file:
```
trace_geodesics   1
trace_stride      10
trace_maxsteps    10000
trace_output      geodesics.h5
```
