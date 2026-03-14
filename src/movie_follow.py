#!/usr/bin/env python
"""
Render a movie where the camera rides along a chosen null geodesic (photon
trajectory), always looking at the black hole, while all geodesics
progressively build up.

Usage:
    python movie_follow.py [dump] --follow INDEX [options]

Arguments:
    dump                GRMHD HDF5 dump file (default: data/dump_SANE.h5)
    --geodesics PATH    Geodesic trajectory HDF5 file (default: output/geodesics.h5)
    --follow INDEX      (required) 0-based index of geodesic to follow
    --n N               Number of geodesics to plot (default: 50)
    --r-max R           Max radius for visualization clipping in r_g (default: 50.0)
    --r-min R           Min radius; camera holds position when geodesic is below (default: 3.0)
    --cam-offset DIST   Perpendicular offset from geodesic path in r_g (default: 2.0)
    --adaptive-fov      Scale camera width with distance from origin
    --fov-scale K       Multiplier for adaptive FOV (default: 2.0)
    --resolution N      Cartesian grid resolution per axis (default: 256)
    --n-frames N        Number of frames (default: derived from max geodesic steps)
    --fps N             Frames per second for output video (default: 30)
    --frame-dir DIR     Directory for output frames (default: frames/)
    --output PATH       Output video path (default: geodesics_movie.mp4)
    --no-video          Generate frames only, skip ffmpeg assembly
    --horizon-alpha A   Horizon sphere alpha 0–1 (0=invisible, 1=opaque; default: 1.0)
    --opacity-multiplier M  Volume opacity multiplier (default: 100)
    --no-density        Skip volume rendering of density
    --window-size N     Frame resolution in pixels (default: 1024)
    --pv-log PATH       Log file for PyVista/VTK output (default: pyvista_warnings.log)

What it does:
    Reads iharm3d GRMHD density on its native MKS grid, interpolates onto a
    uniform Cartesian grid (done once). Camera position is derived from the
    chosen geodesic's trajectory, offset perpendicular to the path direction
    so the camera never sits exactly on the photon track. The followed geodesic
    is highlighted in gold; all others are shown in cyan. Geodesics progressively
    appear as in movie_geodesics_yt.py. Frames are assembled into H.264 MP4.
"""

import argparse
import contextlib
import logging
import os
import signal
import sys
import time
import numpy as np
import pyvista as pv
from tqdm import tqdm

from movie_geodesics_yt import (
    bl_to_cartesian,
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

    - VTK C-level messages are routed to log_path via vtkFileOutputWindow.
    - The 'pyvista' Python logger is redirected to the same file.
    - Returns a one-element list last_msg[0] that always holds the most
      recent message, suitable for display in a tqdm status line.
    """
    import vtk

    last_msg = [""]

    # --- VTK output window → file ---
    vtk_ow = vtk.vtkFileOutputWindow()
    vtk_ow.SetFileName(log_path)
    vtk_ow.FlushOn()
    vtk.vtkOutputWindow.SetInstance(vtk_ow)

    # --- pyvista Python logger → file + last-message capture ---
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
    This catches C-level VTK writes that bypass Python's sys.stderr.
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
# Camera trajectory
# ---------------------------------------------------------------------------

def compute_camera_trajectory(r_all, th_all, ph_all, nsteps, follow_idx,
                               n_frames, max_step, r_max, r_min, offset_dist):
    """
    Compute camera positions along the followed geodesic for each frame.

    The camera is placed at the geodesic point corresponding to the current
    frame, optionally offset perpendicular to the trajectory tangent.
    When the geodesic is below r_min or has ended, the last valid position
    is held.

    Parameters
    ----------
    r_all, th_all, ph_all : ndarray, shape (nph, max_steps)
    nsteps : ndarray, shape (nph,)
    follow_idx : int
        Index into r_all/th_all/ph_all of the followed geodesic.
    n_frames : int
    max_step : int
    r_max : float
    r_min : float
    offset_dist : float
        Perpendicular offset distance; 0 means no offset.

    Returns
    -------
    cam_positions : ndarray, shape (n_frames, 3)
    cam_steps     : ndarray, shape (n_frames,)  — geodesic step index per frame
    """
    ns_follow = int(nsteps[follow_idx])
    r_f  = r_all[follow_idx, :ns_follow]
    th_f = th_all[follow_idx, :ns_follow]
    ph_f = ph_all[follow_idx, :ns_follow]
    x_f, y_f, z_f = bl_to_cartesian(r_f, th_f, ph_f)
    pts = np.stack([x_f, y_f, z_f], axis=1)  # (ns_follow, 3)

    cam_positions = np.zeros((n_frames, 3))
    cam_steps     = np.zeros(n_frames, dtype=int)

    last_valid_pos = pts[0].copy() if ns_follow > 0 else np.array([2.0 * r_max, 0.0, 0.0])

    for t in range(1, n_frames + 1):
        frame_idx = t - 1
        step = max(1, int(t * max_step / n_frames))
        gs = min(step, ns_follow - 1)
        cam_steps[frame_idx] = gs

        raw_pos = pts[gs]
        r_gs = r_f[gs]

        if r_gs > r_max:
            cam_positions[frame_idx] = last_valid_pos
            continue

        if offset_dist > 0 and gs >= 1:
            tangent = raw_pos - pts[gs - 1]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-10:
                tangent /= tangent_norm
            else:
                tangent = np.array([1.0, 0.0, 0.0])

            look = np.array([0.0, 0.0, 0.0]) - raw_pos
            look_norm = np.linalg.norm(look)
            if look_norm > 1e-10:
                look /= look_norm
            else:
                look = np.array([-1.0, 0.0, 0.0])

            perp = np.cross(tangent, look)
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 1e-10:
                perp /= perp_norm
            else:
                perp = np.cross(tangent, [0.0, 0.0, 1.0])
                pn = np.linalg.norm(perp)
                perp = perp / pn if pn > 1e-10 else np.array([0.0, 1.0, 0.0])

            pos = raw_pos + offset_dist * perp
        else:
            pos = raw_pos

        dist = np.linalg.norm(pos)
        if dist < r_min and dist > 1e-10:
            pos = pos * (r_min / dist)

        cam_positions[frame_idx] = pos
        last_valid_pos = pos

    return cam_positions, cam_steps


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
    parser.add_argument("--follow", type=int, required=True,
                        help="(required) 0-based index of geodesic to follow")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of geodesics to plot (default: 50)")
    parser.add_argument("--r-max", type=float, default=50.0,
                        help="Max radius for visualization clipping in r_g (default: 50.0)")
    parser.add_argument("--r-min", type=float, default=3.0,
                        help="Min radius; camera holds position when below (default: 3.0)")
    parser.add_argument("--cam-offset", type=float, default=2.0, metavar="DIST",
                        help="Perpendicular offset from geodesic path in r_g (default: 0.5)")
    parser.add_argument("--adaptive-fov", action="store_true",
                        help="Scale camera width with distance from origin")
    parser.add_argument("--fov-scale", type=float, default=2.0, metavar="K",
                        help="Multiplier for adaptive FOV (default: 2.0)")
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
    parser.add_argument("--slow-frame-threshold", type=float, default=8.0, metavar="SECS",
                        help="Stop early if a frame takes longer than this many seconds (default: 8.0)")
    args = parser.parse_args()

    last_pv_msg = setup_pv_logging(args.pv_log)

    r_max      = args.r_max
    follow_idx = args.follow

    # --- Load and interpolate density (once) ---
    print(f"Loading GRMHD density from {args.dump} ...")
    rho, grid_params = load_grmhd_density(args.dump, r_max)

    print(f"Interpolating to {args.resolution}^3 Cartesian grid ...")
    rho_cart, bbox = interpolate_to_cartesian(rho, grid_params, r_max, args.resolution)

    # --- Build PyVista grid and BH sphere (once) ---
    print("Building PyVista grid ...")
    grid = rho_to_pyvista_grid(rho_cart, bbox, r_max)
    bh_mesh = make_bh_sphere(grid_params["a"])

    # Compute log-density bounds for volume rendering (done once)
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

    # Validate follow index
    import h5py
    with h5py.File(args.geodesics, "r") as f:
        nph_total = int(f["nph"][()])
    if follow_idx < 0 or follow_idx >= nph_total:
        print(f"Error: --follow {follow_idx} out of range [0, {nph_total - 1}]")
        sys.exit(1)

    # Ensure followed geodesic is in the selection
    if follow_idx not in idx:
        idx = np.append(idx, follow_idx)
        print(f"  Added follow_idx={follow_idx} to selection (now {len(idx)} geodesics)")

    max_step = int(nsteps[idx].max())
    n_frames = args.n_frames if args.n_frames is not None else max_step
    print(f"  {len(idx)} geodesics, max steps = {max_step}, frames = {n_frames}")
    print(f"  Following geodesic index {follow_idx} (nsteps={int(nsteps[follow_idx])})")

    # --- Compute camera trajectory (once) ---
    print("Computing camera trajectory ...")
    cam_positions, cam_steps = compute_camera_trajectory(
        r_all, th_all, ph_all, nsteps, follow_idx,
        n_frames, max_step, r_max, args.r_min, args.cam_offset,
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

    ns_follow = int(nsteps[follow_idx])

    for t in range(1, n_frames + 1):
        frame_idx  = t - 1
        step       = max(1, int(t * max_step / n_frames))
        frame_path = os.path.join(args.frame_dir, f"frame_{t:04d}.png")

        # Stop 10 steps after the followed geodesic ends
        if step > ns_follow + 10:
            pbar.close()
            warn_bar.close()
            print(f"\nFollowed geodesic ended at step {ns_follow}; "
                  f"stopping at frame {t} (step {step}).")
            stopped_early = True
            break

        # Build geodesic polydata at current step
        cyan_lines, gold_lines = geodesics_to_polydata_at_step(
            r_all, th_all, ph_all, nsteps, idx, r_max, step,
            follow_idx=follow_idx,
        )

        frame_t0 = time.time()

        def _alarm_handler(signum, frame):
            raise TimeoutError()

        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(slow_threshold) + 1)
        try:
            with _silence_stderr(args.pv_log):
                # Fresh off-screen plotter per frame
                plotter = pv.Plotter(off_screen=True, window_size=[win, win])
                plotter.set_background("black")

                # Volume rendering
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

                # BH sphere
                plotter.add_mesh(bh_mesh, color="#111118", opacity=args.horizon_alpha,
                                 smooth_shading=True)

                # Geodesic tubes
                if cyan_lines is not None:
                    plotter.add_mesh(cyan_lines.tube(radius=0.035), color=CYAN,
                                     opacity=0.7, smooth_shading=True)
                if gold_lines is not None:
                    plotter.add_mesh(gold_lines.tube(radius=0.07), color=GOLD,
                                     opacity=0.95, smooth_shading=True)

                # Camera
                cam_pos = cam_positions[frame_idx]
                plotter.camera.position    = tuple(cam_pos)
                plotter.camera.focal_point = (0.0, 0.0, 0.0)

                xy_dist    = np.sqrt(cam_pos[0]**2 + cam_pos[1]**2)
                total_dist = np.linalg.norm(cam_pos)
                if total_dist > 0 and xy_dist / total_dist < 0.01:
                    plotter.camera.up = (0.01, 1.0, 0.0)
                else:
                    plotter.camera.up = (0.0, 0.0, 1.0)

                plotter.screenshot(frame_path)
                plotter.close()
        except TimeoutError:
            signal.alarm(0)
            frame_elapsed = time.time() - frame_t0
            pbar.close()
            warn_bar.close()
            print(f"\nWARNING: frame {t} timed out after {frame_elapsed:.1f}s "
                  f"(threshold={slow_threshold:.1f}s). Stopping early.")
            stopped_early = True
            break
        finally:
            signal.alarm(0)

        frame_elapsed = time.time() - frame_t0
        frame_times.append(frame_elapsed)

        pbar.set_postfix({"s/frame": f"{frame_elapsed:.1f}"})
        pbar.update(1)
        if last_pv_msg[0]:
            warn_bar.set_description(f"[pv] {last_pv_msg[0]}")

    if not stopped_early:
        pbar.close()
        warn_bar.close()
    print(f"{'Early stop' if stopped_early else 'All frames'} rendered in "
          f"{time.time() - t0:.0f}s ({len(frame_times)} frames)")

    # --- Assemble video ---
    if not args.no_video:
        assemble_video(args.frame_dir, args.output, args.fps)


if __name__ == "__main__":
    main()
