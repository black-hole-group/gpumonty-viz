#!/usr/bin/env python
"""
Render a geodesic movie where the camera flies along a smooth arc above the
midplane, always pointing at the black hole (flyby mode).

Usage:
    python movie_flyby.py [dump] [options]

Arguments:
    dump                    GRMHD HDF5 dump file (default: data/dump_SANE.h5)
    --geodesics PATH        Geodesic trajectory HDF5 file (default: output/geodesics.h5)
    --elevation DEG         Camera elevation above midplane in degrees (default: 30; clamped 0–89)
    --azimuth-start DEG     Starting azimuth in degrees (default: 0)
    --azimuth-sweep DEG     Total azimuthal arc in degrees (default: 180)
    --cam-distance R        Camera distance from origin in r_g (default: 2*r_max)
    --follow INDEX          Optional: highlight one geodesic in gold (camera still flies)
    --n N                   Number of geodesics to plot (default: 50)
    --r-max R               Max radius for visualization clipping in r_g (default: 50.0)
    --resolution N          Cartesian grid resolution per axis (default: 256)
    --n-frames N            Number of frames (default: derived from max geodesic steps)
    --fps N                 Frames per second for output video (default: 30)
    --frame-dir DIR         Directory for output frames (default: frames/)
    --output PATH           Output video path (default: geodesics_movie.mp4)
    --no-video              Generate frames only, skip ffmpeg assembly
    --horizon-alpha A       Horizon sphere opacity 0–1 (default: 1.0)
    --opacity-multiplier M  Volume opacity multiplier (default: 100)
    --no-density            Skip volume rendering of density
    --window-size N         Frame resolution in pixels (default: 1024)
    --tube-radius R         Base geodesic tube radius in r_g (default: 0.07; gold is 2x)
    --pv-log PATH           Log file for PyVista/VTK output (default: pyvista_warnings.log)
    --slow-frame-threshold  Stop if a frame exceeds this many seconds (default: 8.0)

What it does:
    Reads iharm3d GRMHD density on its native MKS grid, interpolates onto a
    uniform Cartesian grid (done once). The camera traces a circular arc at
    constant elevation above the midplane, sweeping from azimuth-start by
    azimuth-sweep degrees, always looking at the origin (black hole).
    Geodesics progressively appear frame-by-frame. Optionally one geodesic is
    highlighted in gold via --follow. Frames are assembled into H.264 MP4.
"""

import argparse
import contextlib
import logging
import os
import sys
import time
import numpy as np
import pyvista as pv
from tqdm import tqdm

from data_utils import (
    load_grmhd_density,
    interpolate_to_cartesian,
    load_geodesics,
    assemble_video,
)
from plot_geodesics_pv import (
    rho_to_pyvista_grid,
    geodesics_to_polydata_at_step,
    make_bh_sphere,
)


# ---------------------------------------------------------------------------
# PyVista / VTK logging capture
# ---------------------------------------------------------------------------

def setup_pv_logging(log_path):
    """
    Redirect PyVista and VTK output to log_path; keep terminal clean.

    Returns a one-element list last_msg[0] that always holds the most
    recent message, suitable for display in a tqdm status line.
    """
    import vtk

    last_msg = [""]

    vtk_ow = vtk.vtkFileOutputWindow()
    vtk_ow.SetFileName(log_path)
    vtk_ow.FlushOn()
    vtk.vtkOutputWindow.SetInstance(vtk_ow)

    class _Capture(logging.Handler):
        def emit(self, record):
            last_msg[0] = self.format(record)

    pv_logger = logging.getLogger("pyvista")
    pv_logger.propagate = False
    pv_logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    pv_logger.addHandler(fh)

    ch = _Capture()
    ch.setLevel(logging.DEBUG)
    pv_logger.addHandler(ch)
    pv_logger.setLevel(logging.DEBUG)

    return last_msg


@contextlib.contextmanager
def _silence_stderr(log_path):
    """
    Redirect OS-level stderr (fd 2) to log_path for the duration of the block.
    Catches C-level VTK writes that bypass Python's sys.stderr.
    """
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    try:
        with open(log_path, "a") as f:
            os.dup2(f.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


# ---------------------------------------------------------------------------
# Flyby camera trajectory
# ---------------------------------------------------------------------------

def compute_flyby_trajectory(n_frames, cam_distance, elevation_deg,
                              azimuth_start_deg, azimuth_sweep_deg):
    """
    Compute camera positions along a circular arc at constant elevation.

    The camera sweeps from azimuth_start_deg by azimuth_sweep_deg degrees
    at a fixed distance and elevation above the midplane, always aimed at
    the origin.

    Parameters
    ----------
    n_frames : int
    cam_distance : float
        Distance from origin in r_g.
    elevation_deg : float
        Degrees above the midplane (clamped to 0–89).
    azimuth_start_deg : float
        Starting azimuth angle in degrees.
    azimuth_sweep_deg : float
        Total arc to sweep in degrees (can be negative for reverse sweep).

    Returns
    -------
    cam_positions : ndarray, shape (n_frames, 3)
    """
    elevation_deg = float(np.clip(elevation_deg, 0.0, 89.0))
    elev = np.radians(elevation_deg)
    t = np.linspace(0.0, 1.0, n_frames)
    phi = np.radians(azimuth_start_deg) + t * np.radians(azimuth_sweep_deg)

    x = cam_distance * np.cos(elev) * np.cos(phi)
    y = cam_distance * np.cos(elev) * np.sin(phi)
    z = np.full(n_frames, cam_distance * np.sin(elev))

    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dump", nargs="?", default="data/dump_SANE.h5",
                        help="GRMHD HDF5 dump file (default: data/dump_SANE.h5)")
    parser.add_argument("--geodesics", default="output/geodesics.h5",
                        help="Geodesic trajectory file (default: output/geodesics.h5)")
    parser.add_argument("--elevation", type=float, default=30.0, metavar="DEG",
                        help="Camera elevation above midplane in degrees (default: 30; clamped 0–89)")
    parser.add_argument("--azimuth-start", type=float, default=0.0, metavar="DEG",
                        help="Starting azimuth in degrees (default: 0)")
    parser.add_argument("--azimuth-sweep", type=float, default=180.0, metavar="DEG",
                        help="Total azimuthal arc in degrees (default: 180)")
    parser.add_argument("--cam-distance", type=float, default=None, metavar="R",
                        help="Camera distance from origin in r_g (default: 2*r_max)")
    parser.add_argument("--follow", type=int, default=None,
                        help="Optional: 0-based index of geodesic to highlight in gold")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of geodesics to plot (default: 50)")
    parser.add_argument("--r-max", type=float, default=50.0,
                        help="Max radius for visualization clipping in r_g (default: 50.0)")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Cartesian grid resolution per axis (default: 256)")
    parser.add_argument("--n-frames", type=int, default=None,
                        help="Number of frames (default: max geodesic steps)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for output video (default: 30)")
    parser.add_argument("--frame-dir", default="frames",
                        help="Directory for output frames (default: frames/)")
    parser.add_argument("--output", default="geodesics_movie.mp4",
                        help="Output video path (default: geodesics_movie.mp4)")
    parser.add_argument("--no-video", action="store_true",
                        help="Generate frames only, skip ffmpeg assembly")
    parser.add_argument("--horizon-alpha", type=float, default=1.0,
                        help="Horizon sphere opacity (0=invisible, 1=opaque, default: 1.0)")
    parser.add_argument("--opacity-multiplier", type=float, default=100.0,
                        help="Volume opacity multiplier (default: 100.0)")
    parser.add_argument("--no-density", action="store_true",
                        help="Skip volume rendering of density")
    parser.add_argument("--window-size", type=int, default=1024,
                        help="Frame resolution in pixels (default: 1024)")
    parser.add_argument("--pv-log", default="pyvista_warnings.log",
                        help="Log file for PyVista/VTK output (default: pyvista_warnings.log)")
    parser.add_argument("--tube-radius", type=float, default=0.07,
                        help="Base tube radius for geodesic lines in r_g (default: 0.07; gold tubes are 2x this)")
    parser.add_argument("--slow-frame-threshold", type=float, default=8.0, metavar="SECS",
                        help="Stop early if a frame takes longer than this many seconds (default: 8.0)")
    args = parser.parse_args()

    last_pv_msg = setup_pv_logging(args.pv_log)

    r_max = args.r_max

    # --- Load and interpolate density (once) ---
    print(f"Loading GRMHD density from {args.dump} ...")
    rho, grid_params = load_grmhd_density(args.dump, r_max)

    print(f"Interpolating to {args.resolution}^3 Cartesian grid ...")
    rho_cart, bbox = interpolate_to_cartesian(rho, grid_params, r_max, args.resolution)

    # --- Build PyVista grid and BH sphere (once) ---
    print("Building PyVista grid ...")
    grid = rho_to_pyvista_grid(rho_cart, bbox, r_max)
    bh_mesh = make_bh_sphere(grid_params["a"])

    density_render = not args.no_density
    if density_render:
        log_rho = grid.point_data["log_density"]
        valid = log_rho[log_rho > np.log10(10 * 1e-30)]
        log_min = float(valid.min()) if valid.size > 0 else -10.0
        log_max = float(log_rho.max())
        opacity_unit = grid.length / (np.mean(grid.dimensions) - 1) * args.opacity_multiplier
        print(f"  log10 density range for TF: [{log_min:.2f}, {log_max:.2f}]")

    # --- Load geodesics ---
    print(f"Loading geodesics from {args.geodesics} ...")
    try:
        r_all, th_all, ph_all, nsteps, idx = load_geodesics(args.geodesics, args.n)
    except FileNotFoundError:
        print(f"Error: geodesic file not found: {args.geodesics}")
        sys.exit(1)

    follow_idx = args.follow
    if follow_idx is not None:
        import h5py
        with h5py.File(args.geodesics, "r") as f:
            nph_total = int(f["nph"][()])
        if follow_idx < 0 or follow_idx >= nph_total:
            print(f"Error: --follow {follow_idx} out of range [0, {nph_total - 1}]")
            sys.exit(1)
        if follow_idx not in idx:
            idx = np.append(idx, follow_idx)
            print(f"  Added follow_idx={follow_idx} to selection (now {len(idx)} geodesics)")

    max_step = int(nsteps[idx].max())
    n_frames = args.n_frames if args.n_frames is not None else max_step
    print(f"  {len(idx)} geodesics, max steps = {max_step}, frames = {n_frames}")

    # --- Compute flyby camera trajectory (once) ---
    cam_distance = args.cam_distance if args.cam_distance is not None else 2.0 * r_max
    print(f"Camera flyby: elevation={args.elevation}°, azimuth {args.azimuth_start}°→"
          f"{args.azimuth_start + args.azimuth_sweep}°, distance={cam_distance:.1f} r_g")
    cam_positions = compute_flyby_trajectory(
        n_frames, cam_distance, args.elevation,
        args.azimuth_start, args.azimuth_sweep,
    )

    # --- Render frames ---
    os.makedirs(args.frame_dir, exist_ok=True)
    win = args.window_size
    print(f"Rendering {n_frames} frames to {args.frame_dir}/ ...")
    t0 = time.time()

    CYAN = (0.0, 1.0, 0.88)
    GOLD = (1.0, 0.85, 0.0)

    pbar = tqdm(total=n_frames, position=0, leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                desc="frame")
    warn_bar = tqdm(total=0, position=1, leave=True, bar_format="{desc}",
                    desc="[pv] (no messages yet)")

    slow_threshold = args.slow_frame_threshold
    frame_times = []
    stopped_early = False

    for t in range(1, n_frames + 1):
        frame_idx  = t - 1
        step       = max(1, int(t * max_step / n_frames))
        frame_path = os.path.join(args.frame_dir, f"frame_{t:04d}.png")

        cyan_lines, gold_lines = geodesics_to_polydata_at_step(
            r_all, th_all, ph_all, nsteps, idx, r_max, step,
            follow_idx=follow_idx,
        )

        frame_t0 = time.time()

        with _silence_stderr(args.pv_log):
            plotter = pv.Plotter(off_screen=True, window_size=[win, win])
            plotter.set_background("black")

            if density_render:
                plotter.add_volume(
                    grid,
                    scalars="log_density",
                    cmap="inferno",
                    opacity="linear",
                    opacity_unit_distance=opacity_unit,
                    clim=[log_min, log_max],
                    shade=False,
                )

            plotter.add_mesh(bh_mesh, color="#111118", opacity=args.horizon_alpha,
                             smooth_shading=True)

            if cyan_lines is not None:
                plotter.add_mesh(cyan_lines.tube(radius=args.tube_radius), color=CYAN,
                                 opacity=0.7, smooth_shading=True)
            if gold_lines is not None:
                plotter.add_mesh(gold_lines.tube(radius=2.0 * args.tube_radius), color=GOLD,
                                 opacity=0.95, smooth_shading=True)

            cam_pos = cam_positions[frame_idx]
            plotter.camera.position    = tuple(cam_pos)
            plotter.camera.focal_point = (0.0, 0.0, 0.0)

            # Per-frame up-vector: avoid gimbal lock when nearly along z-axis
            xy_dist    = np.sqrt(cam_pos[0]**2 + cam_pos[1]**2)
            total_dist = np.linalg.norm(cam_pos)
            if total_dist > 0 and xy_dist / total_dist < 0.01:
                plotter.camera.up = (0.01, 1.0, 0.0)
            else:
                plotter.camera.up = (0.0, 0.0, 1.0)

            plotter.screenshot(frame_path)
            plotter.close()

        frame_elapsed = time.time() - frame_t0
        frame_times.append(frame_elapsed)

        if frame_elapsed > slow_threshold:
            pbar.close()
            warn_bar.close()
            print(f"\nWARNING: frame {t} took {frame_elapsed:.1f}s "
                  f"(threshold={slow_threshold:.1f}s). Stopping early.")
            stopped_early = True
            break

        pbar.set_postfix({"s/frame": f"{frame_elapsed:.1f}"})
        pbar.update(1)
        if last_pv_msg[0]:
            warn_bar.set_description(f"[pv] {last_pv_msg[0]}")

    if not stopped_early:
        pbar.close()
        warn_bar.close()
    print(f"{'Early stop' if stopped_early else 'All frames'} rendered in "
          f"{time.time() - t0:.0f}s ({len(frame_times)} frames)")

    if not args.no_video:
        assemble_video(args.frame_dir, args.output, args.fps)


if __name__ == "__main__":
    main()
