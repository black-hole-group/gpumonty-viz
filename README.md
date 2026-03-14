# GPUMonty Visualization Suite

Visualization tools for photon geodesic trajectories in Kerr spacetime, with optional GRMHD density field overlays. Post-processes output from [GPUmonty](../gpumonty/), a CUDA-based Monte Carlo radiative transfer code for black hole accretion flows.

![Geodesics + density volume render](geodesics_yt.png)

## Quick Start

(1) Go to GPUmonty’s `track` branch:

    git checkout track

(2) Generate geodesics with GPUmonty by adding to your `.par` file:

```
Ns                1000
trace_geodesics   1
trace_stride      10
trace_maxsteps    10000
trace_output      geodesics.h5
```

(3) Visualize:

```bash
# Interactive 3D viewer (Plotly)
python src/plot_geodesics.py output/geodesics.h5 --n 100

# Volume render density + geodesic overlay → PNG (yt)
python src/plot_geodesics_yt.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --n 50

# Movie: geodesics building up step-by-step → MP4 (yt + ffmpeg)
python src/movie_geodesics_yt.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --n 50 --n-frames 100

# Camera rides along a photon trajectory → MP4 (PyVista + ffmpeg)
python src/movie_follow.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --follow 0 --n 50

# Interactive camera exploration (Jupyter + PyVista/trame)
jupyter notebook src/interactive_camera_pv.ipynb
```

## Scripts

| Script | Backend | Description |
|--------|---------|-------------|
| `plot_geodesics.py` | Plotly | Interactive 3D viewer of photon trajectories |
| `plot_geodesics_yt.py` | yt | Volume render of GRMHD density + geodesic overlay (static PNG) |
| `plot_geodesics_pv.py` | PyVista | Helper module: grid builder, geodesic polylines, BH sphere, plotter assembly |
| `movie_geodesics_yt.py` | yt | Movie of geodesics building up step-by-step; assembles MP4 via ffmpeg |
| `movie_follow.py` | PyVista | Camera-follows-geodesic movie with off-screen rendering |
| `interactive_camera_pv.ipynb` | PyVista/trame | Jupyter notebook for interactive camera exploration |

All scripts live in `src/`.

## Dependencies

| Package | Required by |
|---------|-------------|
| `h5py`, `numpy`, `scipy`, `tqdm` | All scripts |
| `plotly` | `plot_geodesics.py` |
| `yt` | `plot_geodesics_yt.py`, `movie_geodesics_yt.py` |
| `pyvista` | `plot_geodesics_pv.py`, `movie_follow.py`, `interactive_camera_pv.ipynb` |
| `pyvista[jupyter]`, `trame` | `interactive_camera_pv.ipynb` |
| `ffmpeg` (external) | `movie_geodesics_yt.py`, `movie_follow.py` |

## CLI Reference

### `plot_geodesics.py` — Interactive 3D Viewer

```
python src/plot_geodesics.py [geodesics.h5] [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `h5file` | `output/geodesics.h5` | Path to geodesic HDF5 file |
| `--n N` | 50 | Number of geodesics to plot (randomly sampled) |
| `--no-bh` | — | Skip drawing the black hole horizon sphere |
| `--cam-dist D` | 5 | Initial camera distance from origin in r_g |

Trajectories are drawn as fluorescent cyan lines on a dark background. An opaque black sphere at r = 1 r_g marks the event horizon.

### `plot_geodesics_yt.py` — Density + Geodesic Volume Render

```
python src/plot_geodesics_yt.py [dump] [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `dump` | `data/dump_SANE.h5` | GRMHD HDF5 dump file |
| `--geodesics PATH` | `output/geodesics.h5` | Geodesic trajectory file |
| `--n N` | 50 | Number of geodesics to plot |
| `--r-max R` | 50.0 | Max radius for visualization clipping (r_g) |
| `--resolution N` | 256 | Cartesian grid resolution per axis |
| `--no-geodesics` | — | Render density only |
| `--output PATH` | `geodesics_yt.png` | Output image path |
| `--cam-position X Y Z` | `(2*r_max, 0, 0)` | Camera position in r_g |

The density is read on its native MKS grid, interpolated to a uniform Cartesian grid via `scipy.ndimage.map_coordinates`, loaded into yt, and volume-rendered. Geodesics are overlaid as `LineSource` objects.

### `movie_geodesics_yt.py` — Geodesic Evolution Movie

```
python src/movie_geodesics_yt.py [dump] [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `dump` | `data/dump_SANE.h5` | GRMHD HDF5 dump file |
| `--geodesics PATH` | `output/geodesics.h5` | Geodesic trajectory file |
| `--n N` | 50 | Number of geodesics to plot |
| `--r-max R` | 50.0 | Max radius (r_g) |
| `--resolution N` | 256 | Cartesian grid resolution |
| `--cam-position X Y Z` | `(2*r_max, 0, 0)` | Camera position in r_g |
| `--n-frames N` | max geodesic steps | Number of output frames |
| `--fps N` | 30 | Video framerate |
| `--frame-dir DIR` | `frames/` | Directory for frame PNGs |
| `--output PATH` | `geodesics_movie.mp4` | Output video path |
| `--no-video` | — | Generate frames only, skip ffmpeg |
| `--yt-log PATH` | `yt_warnings.log` | Capture yt output to file |

Each frame truncates geodesics at step t, creating a progressive build-up effect. Output uses `-pix_fmt yuv420p` for macOS QuickTime compatibility.

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
- **Two rendering backends**: yt provides classical volume rendering with `ColorTransferFunction`; PyVista/VTK enables advanced camera work (following geodesics, interactive exploration).
- **Progressive revelation**: Movie scripts truncate geodesics at increasing step numbers, creating a dramatic build-up of photon paths.
- **Off-screen rendering**: PyVista scripts use `pv.Plotter(off_screen=True)` + `plotter.screenshot()` for headless batch frame generation.
- **`plot_geodesics_pv.py` is a library**: Imported by `movie_follow.py` and `interactive_camera_pv.ipynb`; not run standalone.
- **`movie_follow.py` imports from `movie_geodesics_yt.py`**: Reuses `load_grmhd_density`, `interpolate_to_cartesian`, `load_geodesics`, `assemble_video`, and `bl_to_cartesian`.
