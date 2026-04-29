#!/usr/bin/env python3
import os
import csv
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from casadi import interpolant

import viz


# ---------------------------------------------------------------------
# Track loading aligned with C++ RLManager::load_from_csv()
# ---------------------------------------------------------------------

def read_xy_csv(path):
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :2].astype(float)


def arclength(P):
    if P.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.zeros(P.shape[0], dtype=float)
    if len(d) > 0:
        s[1:] = np.cumsum(d)
    return s


def compute_unit_derivatives_closed(center):
    n = center.shape[0]
    deriv = np.zeros_like(center, dtype=float)
    if n == 0:
        return deriv
    for i in range(n):
        j = (i + 1) % n
        dx = center[j, 0] - center[i, 0]
        dy = center[j, 1] - center[i, 1]
        nrm = np.hypot(dx, dy) + 1e-9
        deriv[i, 0] = dx / nrm
        deriv[i, 1] = dy / nrm
    return deriv


class PlaybackMPC:
    """
    Minimal playback object matching the *current* C++ behavior.

    Important: main.cpp no longer applies symbolic corridor shifting before solve().
    The solver uses the active base corridor and computes numeric corridor bounds
    from right_points/left_points along the warm start. Therefore playback must
    not invent an obstacle-based shifted LUT here.
    """
    def __init__(self, td):
        self.left_lut_x = td["l_lut_x"]
        self.left_lut_y = td["l_lut_y"]
        self.right_lut_x = td["r_lut_x"]
        self.right_lut_y = td["r_lut_y"]



def build_track_only(track_folder):
    center = read_xy_csv(os.path.join(track_folder, "centerline_waypoints.csv"))
    right = read_xy_csv(os.path.join(track_folder, "right_waypoints.csv"))
    left = read_xy_csv(os.path.join(track_folder, "left_waypoints.csv"))

    deriv_file = os.path.join(track_folder, "center_spline_derivatives.csv")
    if os.path.exists(deriv_file):
        deriv = read_xy_csv(deriv_file)
    else:
        deriv = compute_unit_derivatives_closed(center)

    # Match RLManager.cpp extension logic exactly.
    ext_start = 1
    ext_end = 1 + center.shape[0] // 2

    center_ext = np.vstack([center, center[ext_start:ext_end]])
    right_ext = np.vstack([right, right[ext_start:ext_end]])
    left_ext = np.vstack([left, left[ext_start:ext_end]])
    deriv_ext = np.vstack([deriv, deriv[ext_start:ext_end]])

    s_orig = arclength(center)
    s_ext = arclength(center_ext)
    s_total = float(s_orig[-1]) if s_orig.size else 0.0

    td = {
        "center": center,
        "left": left,
        "right": right,
        "center_ext": center_ext,
        "left_ext": left_ext,
        "right_ext": right_ext,
        "deriv_ext": deriv_ext,
        "s_orig": s_orig,
        "s_ext": s_ext,
        "s_total": s_total,
        "c_lut_x": interpolant("c_x_pb", "bspline", [s_ext], center_ext[:, 0]),
        "c_lut_y": interpolant("c_y_pb", "bspline", [s_ext], center_ext[:, 1]),
        "c_lut_dx": interpolant("c_dx_pb", "bspline", [s_ext], deriv_ext[:, 0]),
        "c_lut_dy": interpolant("c_dy_pb", "bspline", [s_ext], deriv_ext[:, 1]),
        "l_lut_x": interpolant("l_x_pb", "bspline", [s_ext], left_ext[:, 0]),
        "l_lut_y": interpolant("l_y_pb", "bspline", [s_ext], left_ext[:, 1]),
        "r_lut_x": interpolant("r_x_pb", "bspline", [s_ext], right_ext[:, 0]),
        "r_lut_y": interpolant("r_y_pb", "bspline", [s_ext], right_ext[:, 1]),
    }

    return td, PlaybackMPC(td)


# ---------------------------------------------------------------------
# Existing CSV loaders
# ---------------------------------------------------------------------

def find_results_dir_with_csv(base_results_dir, required_file="states_ctrls.csv"):
    if not os.path.isdir(base_results_dir):
        raise FileNotFoundError(f"Results directory does not exist: {base_results_dir}")

    direct_file = os.path.join(base_results_dir, required_file)
    if os.path.exists(direct_file):
        return base_results_dir

    candidate_dirs = []
    for d in os.listdir(base_results_dir):
        full = os.path.join(base_results_dir, d)
        if os.path.isdir(full):
            candidate_file = os.path.join(full, required_file)
            if os.path.exists(candidate_file):
                candidate_dirs.append(full)

    if not candidate_dirs:
        raise FileNotFoundError(
            f"Could not find '{required_file}' directly in '{base_results_dir}' "
            f"or in any subdirectory."
        )

    return max(candidate_dirs, key=os.path.getmtime)


def load_main_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if len(rows) == 0:
        raise RuntimeError(f"No rows found in {path}")

    K = len(rows)

    states = np.zeros((K, 5), dtype=float)
    ctrls = np.zeros((K, 3), dtype=float)
    cost_hist = []
    solve_time = []
    v_ref = np.zeros(K, dtype=float)
    logged_limits = {
        "theta_min": None,
        "theta_max": None,
        "vx_min": None,
        "vx_max": None,
        "D_min": None,
        "D_max": None,
        "vs_min": None,
        "vs_max": None,
    }

    def maybe_float(v):
        if v is None:
            return None
        s = str(v).strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None

    for i, r in enumerate(rows):
        states[i] = [
            float(r["x"]),
            float(r["y"]),
            float(r["psi"]),
            float(r["s"]),
            float(r["v_state"]),
        ]
        ctrls[i] = [
            float(r["v_cmd"]),
            float(r["theta"]),
            float(r["p_cmd"]),
        ]
        cost_hist.append(float(r["cost"]) if r["cost"] != "" else np.nan)
        solve_time.append(float(r["solve_time"]) if r["solve_time"] != "" else np.nan)
        v_ref[i] = float(r["v_ref"]) if r["v_ref"] != "" else np.nan

        if i == 0:
            for k in logged_limits.keys():
                logged_limits[k] = maybe_float(r.get(k, None))

    dt = float(rows[1]["t"]) - float(rows[0]["t"]) if K > 1 else 0.25
    return states, ctrls, cost_hist, solve_time, v_ref, dt, logged_limits


def load_predictions_csv(path, num_steps):
    pred_map = {}

    if not os.path.exists(path):
        return [np.zeros((0, 4), dtype=float) for _ in range(num_steps)]

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            step = int(r["step"])
            pred_map.setdefault(step, []).append([
                float(r["x"]),
                float(r["y"]),
                float(r["psi"]),
                float(r["s"]),
            ])

    pred_hist = []
    for step in range(num_steps):
        arr = np.array(pred_map.get(step, []), dtype=float)
        if arr.size == 0:
            arr = np.zeros((0, 4), dtype=float)
        pred_hist.append(arr)

    return pred_hist


def load_obstacles_csv(path, num_steps):
    obs_map = {}

    if not os.path.exists(path):
        return [[] for _ in range(num_steps)]

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            step = int(r["step"])
            obs_map.setdefault(step, []).append({
                "x": float(r["x"]),
                "y": float(r["y"]),
                "radius": float(r["radius"]),
            })

    return [obs_map.get(step, []) for step in range(num_steps)]


def load_u_pred_csv(path, num_steps):
    up_map = {}

    if not os.path.exists(path):
        return [np.zeros((0,), dtype=float) for _ in range(num_steps)]

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            step = int(r["step"])
            if "v_pred" in r and r["v_pred"] != "":
                up_map.setdefault(step, []).append(float(r["v_pred"]))
            elif "p_cmd" in r and r["p_cmd"] != "":
                # Backward compatibility with older logs where this was mislabeled.
                up_map.setdefault(step, []).append(float(r["p_cmd"]))

    return [np.array(up_map.get(step, []), dtype=float) for step in range(num_steps)]


def extend_vhist_for_dashboard(states):
    # In this project, ctrls[:,0] is drivetrain command D (not speed).
    # Use logged v_state from states[:,4] for dashboard velocity traces.
    if states.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    v_state = states[:, 4]
    return np.r_[v_state[0], v_state]


def extend_vref_for_dashboard(v_ref):
    if v_ref.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    return np.r_[v_ref[0], v_ref]


def extend_states_for_dashboard(states):
    if states.shape[0] == 0:
        return np.zeros((0, 4), dtype=float)
    states4 = states[:, :4]
    last = states4[-1:, :]
    return np.vstack([states4, last])


def apply_stride(data, stride):
    if stride <= 1:
        return data
    return data[::stride]


def apply_stride_list(data, stride):
    if stride <= 1:
        return data
    return data[::stride]


def parse_args():
    parser = argparse.ArgumentParser(description="Playback MPC dashboard from CSV logs.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing states_ctrls.csv and related files.")
    parser.add_argument("--track-folder", type=str, default=None,
                        help="Track folder containing center/left/right waypoint CSV files.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every n-th frame for playback and export.")
    parser.add_argument("--simple", action="store_true",
                        help="Use the simpler dashboard instead of diagnosis mode.")
    parser.add_argument("--gif", type=str, default=None,
                        help="If set, save the animation as a GIF to this path.")
    parser.add_argument("--fps", type=int, default=10,
                        help="FPS for GIF export.")
    parser.add_argument("--dpi", type=int, default=100,
                        help="DPI for GIF export.")
    parser.add_argument("--gif-max-frames", type=int, default=220,
                        help="Maximum frames for GIF export (auto-increases stride if needed).")
    parser.add_argument("--theta-max", type=float, default=None,
                        help="Override logged steering angle limit [rad] for constraint visualization.")
    parser.add_argument("--d-min", type=float, default=None,
                        help="Override logged drivetrain state lower limit D_min.")
    parser.add_argument("--d-max", type=float, default=None,
                        help="Override logged drivetrain state upper limit D_max.")
    parser.add_argument("--vs-min", type=float, default=None,
                        help="Override logged progress-speed state lower limit vs_min.")
    parser.add_argument("--vs-max", type=float, default=None,
                        help="Override logged progress-speed state upper limit vs_max.")
    parser.add_argument("--vx-min", type=float, default=None,
                        help="Override logged longitudinal speed lower limit.")
    parser.add_argument("--vx-max", type=float, default=None,
                        help="Override logged longitudinal speed upper limit.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open the matplotlib window.")
    return parser.parse_args()


def save_animation_gif(anim, gif_path, fps, dpi):
    gif_path = os.path.abspath(gif_path)
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    print(f"Saving GIF to: {gif_path}")
    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=dpi)
    except Exception as e:
        raise RuntimeError(
            "GIF export failed. Make sure pillow is installed, e.g. 'pip install pillow'."
        ) from e
    print("GIF export finished.")


def align_1d_length(series, target_len):
    arr = np.asarray(series, dtype=float).reshape(-1)
    if target_len <= 0:
        return np.zeros((0,), dtype=float)
    if arr.size == 0:
        return np.full((target_len,), np.nan, dtype=float)
    if arr.size > target_len:
        return arr[:target_len]
    return np.r_[arr, np.full(target_len - arr.size, arr[-1])]


def align_list_length(items, target_len, default_factory):
    out = list(items)
    if len(out) > target_len:
        return out[:target_len]
    if len(out) < target_len:
        out.extend(default_factory() for _ in range(target_len - len(out)))
    return out


def main():
    args = parse_args()
    stride = max(1, int(args.stride))
    simple_mode = bool(args.simple)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    base_results_dir = os.path.join(project_dir, "results")
    default_track_folder = os.path.join(project_dir, "raceline")

    results_dir = args.results_dir or find_results_dir_with_csv(base_results_dir, required_file="states_ctrls.csv")
    track_folder = args.track_folder or default_track_folder

    main_csv = os.path.join(results_dir, "states_ctrls.csv")
    pred_csv = os.path.join(results_dir, "predictions.csv")
    obs_csv = os.path.join(results_dir, "obstacles.csv")
    upred_csv = os.path.join(results_dir, "u_pred.csv")

    print("Using results folder:", results_dir)
    print("Main CSV:", main_csv)
    print("Track folder:", track_folder)

    print("Loading track with C++-compatible RLManager geometry...")
    td, mpc_like = build_track_only(track_folder)

    print("Loading CSV data...")
    states, ctrls, cost_hist, solve_time, v_ref, dt, logged_limits = load_main_csv(main_csv)
    num_steps = ctrls.shape[0]

    pred_hist = load_predictions_csv(pred_csv, num_steps)
    obs_log = load_obstacles_csv(obs_csv, num_steps)
    u_pred_hist = load_u_pred_csv(upred_csv, num_steps)

    states_plot = extend_states_for_dashboard(states)
    v_hist = extend_vhist_for_dashboard(states)
    v_ref_series = extend_vref_for_dashboard(v_ref)

    states_plot = apply_stride(states_plot, stride)
    pred_hist = apply_stride_list(pred_hist, stride)
    obs_log = apply_stride_list(obs_log, stride)
    ctrls = apply_stride(ctrls, stride)
    cost_hist = apply_stride_list(cost_hist, stride)
    solve_time = apply_stride_list(solve_time, stride)
    u_pred_hist = apply_stride_list(u_pred_hist, stride)
    v_hist = apply_stride(v_hist, stride)
    v_ref_series = apply_stride(v_ref_series, stride)
    dt = dt * stride

    T = states_plot.shape[0]

    # For GIF export, auto-decimate long runs to keep export time and file size reasonable.
    if args.gif and T > max(1, args.gif_max_frames):
        auto_stride = int(math.ceil(float(T) / float(args.gif_max_frames)))
        print(f"Auto GIF stride: {auto_stride} (from {T} frames to <= {args.gif_max_frames})")
        states_plot = apply_stride(states_plot, auto_stride)
        pred_hist = apply_stride_list(pred_hist, auto_stride)
        obs_log = apply_stride_list(obs_log, auto_stride)
        ctrls = apply_stride(ctrls, auto_stride)
        cost_hist = apply_stride_list(cost_hist, auto_stride)
        solve_time = apply_stride_list(solve_time, auto_stride)
        u_pred_hist = apply_stride_list(u_pred_hist, auto_stride)
        v_hist = apply_stride(v_hist, auto_stride)
        v_ref_series = apply_stride(v_ref_series, auto_stride)
        dt = dt * auto_stride
        T = states_plot.shape[0]

    v_hist = align_1d_length(v_hist, T)
    v_ref_series = align_1d_length(v_ref_series, T)
    cost_hist = align_1d_length(cost_hist, T)
    solve_time = align_1d_length(solve_time, T)
    pred_hist = align_list_length(pred_hist, T, lambda: np.zeros((0, 4), dtype=float))
    obs_log = align_list_length(obs_log, T, lambda: [])
    u_pred_hist = align_list_length(u_pred_hist, T, lambda: np.zeros((0,), dtype=float))

    print(f"Playback stride = {stride}")
    print(f"Frames after decimation = {T}")
    print("Note: playback uses base corridor LUTs only, matching current C++ solve() flow.")

    limits = {
        "theta_max": logged_limits.get("theta_max"),
        "D_min": logged_limits.get("D_min"),
        "D_max": logged_limits.get("D_max"),
        "vs_min": logged_limits.get("vs_min"),
        "vs_max": logged_limits.get("vs_max"),
        "vx_min": logged_limits.get("vx_min"),
        "vx_max": logged_limits.get("vx_max"),
    }

    if args.theta_max is not None:
        limits["theta_max"] = float(args.theta_max)
    if args.d_min is not None:
        limits["D_min"] = float(args.d_min)
    if args.d_max is not None:
        limits["D_max"] = float(args.d_max)
    if args.vs_min is not None:
        limits["vs_min"] = float(args.vs_min)
    if args.vs_max is not None:
        limits["vs_max"] = float(args.vs_max)
    if args.vx_min is not None:
        limits["vx_min"] = float(args.vx_min)
    if args.vx_max is not None:
        limits["vx_max"] = float(args.vx_max)

    if simple_mode:
        anim, fig = viz.animate_mpc_dashboard(
            td=td,
            states=states_plot,
            pred_hist=pred_hist,
            dt=dt,
            mpc=mpc_like,
            obstacles_log=obs_log,
            v_hist=v_hist,
            v_ref=v_ref_series,
        )
    else:
        anim, fig = viz.animate_mpc_dashboard_diagnosis(
            td=td,
            states=states_plot,
            pred_hist=pred_hist,
            dt=dt,
            mpc=mpc_like,
            obstacles_log=obs_log,
            v_hist=v_hist,
            v_ref=v_ref_series,
            ctrls=ctrls,
            cost_hist=cost_hist,
            solve_time=solve_time,
            lap_s0=0.0,
            show_start_line=True,
            u_pred_hist=u_pred_hist,
            limits=limits,
        )

    if args.gif:
        save_animation_gif(anim, args.gif, fps=args.fps, dpi=args.dpi)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    return anim, fig


if __name__ == "__main__":
    main()
