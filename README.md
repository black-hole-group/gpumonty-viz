# GPUmonty Visualization Suite

Visualization tools for photon geodesic trajectories in Kerr spacetime, with optional GRMHD density field overlays. Post-processes output from [GPUmonty](https://github.com/black-hole-group/gpumonty), a CUDA-based Monte Carlo radiative transfer code for black hole accretion flows. Based on PyVista for rendering and `ffmpeg` for MP4 generation.

![Demo](https://github.com/rsnemmen/rsnemmen.github.io/blob/26e5262260605c88a68d6a8f7fb17b7973bb3e5b/assets/video/movie_200_near-ezgif.com-video-to-webp-converter.webp)
**Movie 1:** Null geodesics generation and propagation from a hot accretion flow around a Kerr black hole. [SANE GRMHD simulation data obtained here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XZECPF).


## Quick Start

(0) Install [PyVista](https://pyvista.org), the rendering engine:

```shell
# conda
conda install -c conda-forge pyvista
# or pip
pip install pyvista
```

(1) Go to GPUmonty's directory, switch to the `track` branch and compile:

```bash
cd gpumonty
git checkout track
make -j <# of CPU cores>
```

(2) Include in the parameter file (e.g. `track.par` file):

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

(3) Run gpumonty to produce an HDF5 file with the geodesic tracks:

```bash
./gpumonty -par track.par
```

Let’s assume the file containing the geodesic tracks produced by gpumonty is named `output/geodesics.h5`.

(4) Visualize the geodesics tracks:

```bash
# Movie: geodesics building up step-by-step → MP4
python src/movie_static.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --n 50

# Camera rides along photon trajectory "10" → MP4
python src/movie_follow.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --follow 10 --n 50

# Camera flies an arc above the midplane → MP4 
python src/movie_flyby.py data/dump_SANE.h5 --geodesics output/geodesics.h5 --n 50

# Interactive camera exploration (Jupyter)
jupyter notebook src/interactive_camera_pv.ipynb
```

## Scripts

| Script | Backend | Description |
|--------|---------|-------------|
| `plot_geodesics_pv.py` | PyVista | Helper module: grid builder, geodesic polylines, BH sphere, plotter assembly |
| `movie_static.py` | PyVista | Movie of geodesics building up step-by-step; assembles MP4 via ffmpeg |
| `movie_follow.py` | PyVista | Camera-follows-geodesic movie with off-screen rendering |
| `movie_flyby.py` | PyVista | Flyby camera arc above midplane, always looking at the black hole |
| `interactive_camera_pv.ipynb` | PyVista/trame | Jupyter notebook for interactive camera exploration |

All scripts live in `src/`.

## Dependencies

| Package | Required by |
|---------|-------------|
| `h5py`, `numpy`, `scipy`, `tqdm` | All scripts |
| `pyvista` | `plot_geodesics_pv.py`, `movie_static.py`, `movie_follow.py`, `movie_flyby.py`, `interactive_camera_pv.ipynb` |
| `pyvista[jupyter]`, `trame` | `interactive_camera_pv.ipynb` |
| `ffmpeg` (external) | `movie_static.py`, `movie_follow.py`, `movie_flyby.py` |

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
| `--look-tangent` | — | Camera looks along geodesic tangent instead of at origin |
| `--look-distance D` | 10.0 | How far ahead along tangent to place focal point (r_g) |
| `--fov DEG` | 30.0 | Camera field of view in degrees |
| `--window-size N` | 1024 | Frame resolution in pixels (square) |

The followed geodesic is highlighted in gold; all others are cyan. The camera offset is computed via cross product of the trajectory tangent and the look-at direction.

### `movie_flyby.py` — Flyby Camera Arc Movie

```
python src/movie_flyby.py [dump] [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `dump` | `data/dump_SANE.h5` | GRMHD HDF5 dump file |
| `--geodesics PATH` | `output/geodesics.h5` | Geodesic trajectory file |
| `--elevation DEG` | 30 | Camera elevation above midplane in degrees (clamped 0–89) |
| `--azimuth-start DEG` | 0 | Starting azimuth in degrees |
| `--azimuth-sweep DEG` | 180 | Total azimuthal arc in degrees |
| `--cam-distance R` | 2×r_max | Camera distance from origin (r_g) |
| `--follow INDEX` | — | Optional: highlight one geodesic in gold |
| `--n N` | 50 | Number of geodesics to plot |
| `--r-max R` | 50.0 | Max radius (r_g) |
| `--resolution N` | 256 | Cartesian grid resolution |
| `--n-frames N` | max geodesic steps | Number of frames |
| `--fps N` | 30 | Framerate |
| `--frame-dir DIR` | `frames/` | Directory for frame PNGs |
| `--output PATH` | `geodesics_movie.mp4` | Output video path |
| `--no-video` | — | Frames only |
| `--horizon-alpha A` | 1.0 | Event horizon sphere opacity (0–1) |
| `--opacity-multiplier M` | 100.0 | Volume rendering opacity multiplier |
| `--no-density` | — | Skip density volume rendering |
| `--window-size N` | 1024 | Frame resolution in pixels (square) |
| `--tube-radius R` | 0.07 | Base geodesic tube radius in r_g (gold is 2×) |
| `--slow-frame-threshold SECS` | 8.0 | Stop early if a frame exceeds this many seconds |

The camera sweeps a circular arc at constant distance and elevation above the midplane, always pointing at the origin. Geodesics progressively build up frame-by-frame as in `movie_static.py`.

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
