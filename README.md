# GPUMonty Visualization Suite

Visualization tools for photon geodesic trajectories in Kerr spacetime, with optional GRMHD density field overlays. Post-processes output from [GPUmonty](../gpumonty/), a CUDA-based Monte Carlo radiative transfer code for black hole accretion flows.

![Geodesics + density volume render](geodesics_yt.png)

## Quick Start

(1) Go to GPUmonty's directory, switch to the `track` branch and compile:

```bash
cd gpumonty
git checkout track
make -j <# of CPU cores>
```

(2) Add to the `.par` parameter file:

```
Ns                1000
trace_geodesics   1
trace_stride      10
trace_maxsteps    10000
trace_output      geodesics.h5
```

| Parameter | Description |
|-----------|-------------|
| `Ns` | Number of photons (superphotons) to sample |
| `trace_geodesics` | Enable geodesic trajectory recording |
| `trace_stride` | Save photon position every N integration steps |
| `trace_maxsteps` | Maximum integration steps per photon |
| `trace_output` | Output HDF5 file for geodesic tracks |

(3) Run the code to produce an HDF5 file w/ geodesic tracks:

```bash
./gpumonty -par track.par
```

(4) Visualize:

```bash
# Movie: geodesics building up step-by-step → MP4 (PyVista + ffmpeg)
python src/movie_static.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --n 50 --n-frames 100

# Camera rides along a photon trajectory → MP4 (PyVista + ffmpeg)
python src/movie_follow.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --follow 0 --n 50

# Interactive camera exploration (Jupyter + PyVista/trame)
jupyter notebook src/interactive_camera_pv.ipynb
```

## Scripts

| Script | Backend | Description |
|--------|---------|-------------|
| `plot_geodesics_pv.py` | PyVista | Helper module: grid builder, geodesic polylines, BH sphere, plotter assembly |
| `movie_static.py` | PyVista | Movie of geodesics building up step-by-step; assembles MP4 via ffmpeg |
| `movie_follow.py` | PyVista | Camera-follows-geodesic movie with off-screen rendering |
| `interactive_camera_pv.ipynb` | PyVista/trame | Jupyter notebook for interactive camera exploration |

All scripts live in `src/`.

## Dependencies

| Package | Required by |
|---------|-------------|
| `h5py`, `numpy`, `scipy`, `tqdm` | All scripts |
| `pyvista` | `plot_geodesics_pv.py`, `movie_static.py`, `movie_follow.py`, `interactive_camera_pv.ipynb` |
| `pyvista[jupyter]`, `trame` | `interactive_camera_pv.ipynb` |
| `ffmpeg` (external) | `movie_static.py`, `movie_follow.py` |

## CLI Reference

### `movie_follow.py` — Camera-Follows-Geodesic Movie

```
python src/movie_follow.py [dump] --follow INDEX [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `dump` | `data/dump_SANE.h5` | GRMHD HDF5 dump file |
| `--geodesics PATH` | `output/geodesics.h5` | Geodesic trajectory file |
| `--follow INDEX` | (required) | 0-based index of geodesic to ride |
| `--n N` | 50 | Number of geodesics to plot |
| `--r-max R` | 50.0 | Max radius (r_g) |
| `--r-min R` | 3.0 | Min radius; camera holds position below this |
| `--cam-offset DIST` | 2.0 | Perpendicular offset from geodesic path (r_g) |
| `--adaptive-fov` | — | Scale camera width with distance from origin |
| `--fov-scale K` | 2.0 | Multiplier for adaptive FOV |
| `--resolution N` | 256 | Cartesian grid resolution |
| `--n-frames N` | max geodesic steps | Number of frames |
| `--fps N` | 30 | Framerate |
| `--frame-dir DIR` | `frames/` | Directory for frame PNGs |
| `--output PATH` | `geodesics_movie.mp4` | Output video path |
| `--no-video` | — | Frames only |
| `--horizon-alpha A` | 1.0 | Event horizon sphere opacity (0–1) |
| `--opacity-multiplier M` | 2.0 | Volume rendering opacity multiplier |
| `--no-density` | — | Skip density volume rendering |
| `--window-size N` | 1024 | Frame resolution in pixels (square) |

The followed geodesic is highlighted in gold; all others are cyan. The camera offset is computed via cross product of the trajectory tangent and the look-at direction.

## Input Data Formats

### `geodesics.h5`

Produced by GPUmonty with `trace_geodesics 1`. Contains photon trajectories in Boyer-Lindquist coordinates:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `r` | `[nph, max_saved]` | Radial coordinate |
| `theta` | `[nph, max_saved]` | Polar angle |
| `phi` | `[nph, max_saved]` | Azimuthal angle |
| `nsteps` | `[nph]` | Valid steps per photon (remaining entries are zero) |
| `nph` | scalar | Number of traced photons |
| `max_saved_steps` | scalar | `trace_maxsteps / trace_stride + 1` |

### GRMHD dump (iharm3d format)

Standard iharm3d HDF5 dump file. The scripts read:
- `/header/geom/mks/hslope` — latitude grid compression parameter
- `/header/geom/startx1..3`, `dx1..3` — MKS grid bounds and spacing
- `/header/n1`, `n2`, `n3` — grid dimensions
- `/prims/` — primitive variables array `[N1, N2, N3, N_PRIM]`; index 0 is density

## ffmpeg Command

To manually assemble frames into an MP4:

```bash
ffmpeg -y -framerate 30 -i frames/frame_%04d.png \
    -c:v libx264 -preset medium -crf 18 \
    -pix_fmt yuv420p \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    geodesics_movie.mp4
```

## Design Notes

- **Coordinate pipeline**: MKS → Boyer-Lindquist → Cartesian. The MKS theta inversion uses a monotonic lookup table (`np.searchsorted` on the theta equation).
- **Single rendering backend**: PyVista/VTK handles all visualization (volume rendering, geodesic tubes, interactive exploration, camera-following).
- **Off-screen rendering**: PyVista scripts use `pv.Plotter(off_screen=True)` + `plotter.screenshot()` for headless batch frame generation.
- **`plot_geodesics_pv.py` is a library**: Imported by `movie_follow.py` and `interactive_camera_pv.ipynb`; not run standalone.
- **Shared data utilities in `data_utils.py`**: All scripts import `load_grmhd_density`, `interpolate_to_cartesian`, `load_geodesics`, `assemble_video`, and `bl_to_cartesian` from here.
