#!/usr/bin/env python
"""
Render a movie of geodesic evolution: photon trajectories build up step-by-step.

Usage:
    python movie_geodesics_yt.py [dump] [options]

Arguments:
    dump                GRMHD HDF5 dump file (default: data/dump_SANE.h5)
    --geodesics PATH    Geodesic trajectory HDF5 file (default: output/geodesics.h5)
    --n N               Number of geodesics to plot (default: 50)
    --r-max R           Max radius for visualization clipping in r_g (default: 50.0)
    --resolution N      Cartesian grid resolution per axis (default: 256)
    --cam-position X Y Z
                        Camera position in r_g (default: along +x at 2*r_max)
    --n-frames N        Number of frames (default: derived from max geodesic steps)
    --fps N             Frames per second for output video (default: 30)
    --frame-dir DIR     Directory for output frames (default: frames/)
    --output PATH       Output video path (default: geodesics_movie.mp4)
    --no-video          Generate frames only, skip ffmpeg assembly

What it does:
    Reads iharm3d GRMHD density on its native MKS grid, interpolates onto a
    uniform Cartesian grid (done once). For each frame t, each geodesic is
    truncated at step min(t, nsteps[i]) so photons progressively appear to
    propagate. Frames are assembled into an H.264 MP4 via ffmpeg with
    yuv420p pixel format for macOS QuickTime compatibility.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import h5py
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import yt
from yt.visualization.volume_rendering.api import LineSource


# ---------------------------------------------------------------------------
# Coordinate helpers  (copied from plot_geodesics_yt.py)
# ---------------------------------------------------------------------------

def bl_to_cartesian(r, theta, phi):
    """Convert Boyer-Lindquist spherical coords to Cartesian."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def mks_to_bl(X1, X2, X3, hslope):
    """Convert MKS coordinates to Boyer-Lindquist (r, theta, phi)."""
    r = np.exp(X1)
    theta = np.pi * X2 + 0.5 * (1.0 - hslope) * np.sin(2.0 * np.pi * X2)
    phi = X3
    return r, theta, phi


# ---------------------------------------------------------------------------
# GRMHD data loading  (copied from plot_geodesics_yt.py)
# ---------------------------------------------------------------------------

def load_grmhd_density(dump_path, r_max):
    """
    Read iharm3d HDF5 dump and return density array with grid parameters.

    Returns
    -------
    rho : ndarray, shape (n1, n2, n3)
    grid_params : dict with keys startx1, dx1, startx2, dx2, startx3, dx3,
                  hslope, a, n1, n2, n3
    """
    with h5py.File(dump_path, "r") as f:
        geom = f["header/geom"]
        mks  = geom["mks"]

        n1 = int(f["header/n1"][()])
        n2 = int(f["header/n2"][()])
        n3 = int(f["header/n3"][()])

        startx1 = float(geom["startx1"][()])
        startx2 = float(geom["startx2"][()])
        startx3 = float(geom["startx3"][()])
        dx1 = float(geom["dx1"][()])
        dx2 = float(geom["dx2"][()])
        dx3 = float(geom["dx3"][()])

        hslope = float(mks["hslope"][()])
        a      = float(mks["a"][()])

        # prims layout: [n1, n2, n3, nprim]; index 0 = RHO
        rho = f["prims"][:, :, :, 0].astype(np.float64)

    grid_params = dict(
        n1=n1, n2=n2, n3=n3,
        startx1=startx1, dx1=dx1,
        startx2=startx2, dx2=dx2,
        startx3=startx3, dx3=dx3,
        hslope=hslope, a=a,
    )
    return rho, grid_params


# ---------------------------------------------------------------------------
# Cartesian interpolation  (copied from plot_geodesics_yt.py)
# ---------------------------------------------------------------------------

def interpolate_to_cartesian(rho, grid_params, r_max, N_cart):
    """
    Interpolate GRMHD density from MKS grid onto a uniform Cartesian cube.

    Returns
    -------
    rho_cart : ndarray, shape (N_cart, N_cart, N_cart)
    bbox : list of [min, max] pairs for each axis
    """
    gp = grid_params
    hslope = gp["hslope"]

    r_in = np.exp(gp["startx1"] + 0.5 * gp["dx1"])
    a = gp["a"]
    r_h = 1.0 + np.sqrt(max(0.0, 1.0 - a**2))
    r_mask_inner = max(r_in, r_h)

    xyz = np.linspace(-r_max, r_max, N_cart)
    Xc, Yc, Zc = np.meshgrid(xyz, xyz, xyz, indexing="ij")

    r_c     = np.sqrt(Xc**2 + Yc**2 + Zc**2)
    theta_c = np.arccos(np.clip(Zc / np.where(r_c > 0, r_c, 1e-30), -1.0, 1.0))
    phi_c   = np.arctan2(Yc, Xc) % (2.0 * np.pi)

    X1_c  = np.log(np.where(r_c > 0, r_c, 1e-30))
    i_frac = (X1_c - gp["startx1"]) / gp["dx1"] - 0.5

    X2_table    = np.linspace(0.0, 1.0, 10000)
    theta_table = np.pi * X2_table + 0.5 * (1.0 - hslope) * np.sin(2.0 * np.pi * X2_table)
    X2_c  = np.interp(theta_c, theta_table, X2_table)
    j_frac = (X2_c - gp["startx2"]) / gp["dx2"] - 0.5

    X3_c  = phi_c
    k_frac = (X3_c - gp["startx3"]) / gp["dx3"] - 0.5

    coords = np.array([i_frac.ravel(), j_frac.ravel(), k_frac.ravel()])
    rho_cart = map_coordinates(rho, coords, order=1, mode="nearest").reshape(N_cart, N_cart, N_cart)

    floor = 1e-30
    mask = (r_c < r_mask_inner) | (r_c > r_max)
    rho_cart[mask] = floor
    rho_cart = np.maximum(rho_cart, floor)

    bbox = [[-r_max, r_max], [-r_max, r_max], [-r_max, r_max]]
    return rho_cart, bbox


# ---------------------------------------------------------------------------
# Geodesic loading  (copied from plot_geodesics_yt.py)
# ---------------------------------------------------------------------------

def load_geodesics(h5path, n_plot):
    """Load geodesic trajectories from GPUmonty HDF5 output."""
    with h5py.File(h5path, "r") as f:
        nph     = int(f["nph"][()])
        r_all   = f["r"][:]
        th_all  = f["theta"][:]
        ph_all  = f["phi"][:]
        nsteps  = f["nsteps"][:].astype(int)

    n_plot = min(n_plot, nph)
    idx = np.random.choice(nph, size=n_plot, replace=False)
    return r_all, th_all, ph_all, nsteps, idx


# ---------------------------------------------------------------------------
# Progressive geodesic segments
# ---------------------------------------------------------------------------

def geodesics_to_line_segments_at_step(r_all, th_all, ph_all, nsteps, idx, r_max, step):
    """
    Convert geodesic data to yt LineSource segment arrays, truncated at `step`.

    Each geodesic i is shown up to min(step, nsteps[i]) points, creating the
    progressive reveal effect.

    Returns
    -------
    segments : ndarray, shape (N_segments, 2, 3), or None
    colors   : ndarray, shape (N_segments, 4) RGBA fluorescent cyan, or None
    """
    seg_list = []
    for i in idx:
        ns = min(step, int(nsteps[i]))
        if ns < 2:
            continue
        r  = r_all[i, :ns]
        th = th_all[i, :ns]
        ph = ph_all[i, :ns]
        x, y, z = bl_to_cartesian(r, th, ph)
        pts = np.stack([x, y, z], axis=1)  # (ns, 3)

        inside = r <= r_max
        for s in range(len(pts) - 1):
            if inside[s] and inside[s + 1]:
                seg_list.append([pts[s], pts[s + 1]])

    if not seg_list:
        return None, None

    segments = np.array(seg_list, dtype=np.float64)
    colors   = np.tile([0.0, 1.0, 0.88, 0.6], (len(seg_list), 1))
    return segments, colors


# ---------------------------------------------------------------------------
# yt logging capture
# ---------------------------------------------------------------------------

def setup_yt_logging(log_path):
    """Suppress yt console output; write everything to log_path; capture latest message."""
    last = [""]

    class _Capture(logging.Handler):
        def emit(self, record):
            last[0] = self.format(record)

    yt_logger = logging.getLogger("yt")
    yt_logger.propagate = False          # suppress yt's own console output
    yt_logger.handlers.clear()           # remove any handlers yt added at import

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    yt_logger.addHandler(fh)

    ch = _Capture()
    ch.setFormatter(logging.Formatter("%(message)s"))
    yt_logger.addHandler(ch)

    return last


# ---------------------------------------------------------------------------
# Scene construction (built once, reused per frame)
# ---------------------------------------------------------------------------

def build_base_scene(ds, r_max, cam_position):
    """
    Create yt scene with volume source and camera configured.
    No LineSource is added here — that is done per frame in render_frame().

    Returns
    -------
    sc : yt Scene object ready for per-frame rendering
    """
    from yt.visualization.volume_rendering.transfer_functions import ColorTransferFunction

    ad = ds.all_data()
    rho_all = np.array(ad[("stream", "density")])
    rho_max = float(rho_all.max())
    floor = 1e-30
    valid = rho_all[rho_all > 10 * floor]
    rho_min_valid = float(valid.min()) if valid.size > 0 else 1e-10
    log_min = np.log10(rho_min_valid)
    log_max = np.log10(max(rho_max, 1e-30))
    print(f"  density range: [{rho_min_valid:.3e}, {rho_max:.3e}]  "
          f"log10 range: [{log_min:.2f}, {log_max:.2f}]")

    tf = ColorTransferFunction((log_min, log_max))
    tf.add_layers(N=30, colormap="inferno", alpha=[0.3]*30, w=0.1)

    sc = yt.create_scene(ds, ("stream", "density"))
    vol = sc[0]
    vol.set_log(True)
    vol.transfer_function = tf

    cam = sc.camera
    cam.set_width(ds.arr(2.0 * r_max, "code_length"))
    cam.position = ds.arr(cam_position, "code_length")
    cam.focus    = ds.arr([0.0, 0.0, 0.0], "code_length")
    cam.north_vector = np.array([0.0, 0.0, 1.0])
    cam.switch_orientation()
    cam.set_resolution((800, 800))

    return sc


def render_frame(sc, segments, colors, frame_path, sigma_clip=4.0):
    """Replace the geodesic LineSource on the scene and save one frame."""
    if segments is not None:
        sc["geodesics"] = LineSource(segments, colors)
    elif "geodesics" in sc.sources:
        del sc.sources["geodesics"]

    sc.save(frame_path, sigma_clip=sigma_clip)


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def assemble_video(frame_dir, output_video, fps):
    """Assemble PNG frames into an H.264 MP4 via ffmpeg."""
    if shutil.which("ffmpeg") is None:
        print("Warning: ffmpeg not found. Skipping video assembly.")
        print(f"  Frames are in: {frame_dir}")
        return

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_video,
    ]
    print(f"Assembling video: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}")
    else:
        print(f"Saved video: {output_video}")


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
    parser.add_argument("--n", type=int, default=50,
                        help="Number of geodesics to plot (default: 50)")
    parser.add_argument("--r-max", type=float, default=50.0,
                        help="Max radius for visualization clipping in r_g (default: 50.0)")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Cartesian grid resolution per axis (default: 256)")
    parser.add_argument("--cam-position", nargs=3, type=float,
                        metavar=("X", "Y", "Z"), default=None,
                        help="Camera position in r_g (default: (2*r_max, 0, 0))")
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
    parser.add_argument("--yt-log", default="yt_warnings.log",
                        help="Log file for yt output (default: yt_warnings.log)")
    args = parser.parse_args()

    last_yt_msg = setup_yt_logging(args.yt_log)

    r_max = args.r_max
    cam_position = args.cam_position if args.cam_position else [2.0 * r_max, 0.0, 0.0]

    # --- Load and interpolate density (once) ---
    print(f"Loading GRMHD density from {args.dump} ...")
    rho, grid_params = load_grmhd_density(args.dump, r_max)

    print(f"Interpolating to {args.resolution}^3 Cartesian grid ...")
    rho_cart, bbox = interpolate_to_cartesian(rho, grid_params, r_max, args.resolution)

    # --- Build yt dataset (once) ---
    data = {"density": (rho_cart, "g/cm**3")}
    bbox_arr = np.array(bbox)
    ds = yt.load_uniform_grid(
        data,
        rho_cart.shape,
        length_unit="code_length",
        bbox=bbox_arr,
        periodicity=(False, False, False),
    )

    # --- Load geodesics ---
    print(f"Loading geodesics from {args.geodesics} ...")
    try:
        r_all, th_all, ph_all, nsteps, idx = load_geodesics(args.geodesics, args.n)
    except FileNotFoundError:
        print(f"Error: geodesic file not found: {args.geodesics}")
        sys.exit(1)

    max_step = int(nsteps[idx].max())
    n_frames = args.n_frames if args.n_frames is not None else max_step
    print(f"  {len(idx)} geodesics, max steps = {max_step}, frames = {n_frames}")

    # --- Build base scene (once) ---
    print("Building yt scene ...")
    sc = build_base_scene(ds, r_max, cam_position)

    # --- Render frames ---
    os.makedirs(args.frame_dir, exist_ok=True)
    print(f"Rendering {n_frames} frames to {args.frame_dir}/ ...")
    t0 = time.time()

    pbar = tqdm(total=n_frames, position=0, leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                desc="frame")
    warn_bar = tqdm(total=0, position=1, leave=True, bar_format="{desc}",
                    desc="[yt] (no messages yet)")

    for t in range(1, n_frames + 1):
        step = max(1, int(t * max_step / n_frames))
        frame_path = os.path.join(args.frame_dir, f"frame_{t:04d}.png")
        segments, colors = geodesics_to_line_segments_at_step(
            r_all, th_all, ph_all, nsteps, idx, r_max, step
        )
        render_frame(sc, segments, colors, frame_path)
        pbar.update(1)
        if last_yt_msg[0]:
            warn_bar.set_description(f"[yt] {last_yt_msg[0]}")

    pbar.close()
    warn_bar.close()
    print(f"All frames rendered in {time.time() - t0:.0f}s")

    # --- Assemble video ---
    if not args.no_video:
        assemble_video(args.frame_dir, args.output, args.fps)


if __name__ == "__main__":
    main()
