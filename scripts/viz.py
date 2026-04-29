#!/usr/bin/env python3
# viz.py
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
#mpl.use('TkAgg') # no UI backend
# ----------------- small helpers -----------------

def plot_boundary_markers(ax, right_pts, left_pts, put_legend=True):
    """Draw one pair of markers and a short normal arrow."""
    i = 0
    rx, ry = right_pts[i]
    lx, ly = left_pts[i]
    tx, ty = (rx - lx), (ry - ly)
    nx, ny = -ty, tx

    ax.scatter([rx], [ry], c='r', marker='^',
               label='Right markers' if put_legend else "_nolegend_")
    ax.scatter([lx], [ly], c='b', marker='^',
               label='Left markers'  if put_legend else "_nolegend_")

    # small normal at midpoint
    mx, my = (lx + rx) * 0.5, (ly + ry) * 0.5
    scale = 0.5
    ax.arrow(mx, my, nx * scale, ny * scale,
             head_width=0.2, length_includes_head=True,
             color='0.25', alpha=0.6,
             label='Boundary normals' if put_legend else "_nolegend_")

def static_plot(td, states, obstacles, r_pts=None, l_pts=None,
                title="MPC Simulated Trajectory"):
    """Simple static figure: track, states, optional obstacles & last corridor markers."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(td['center'][:,0], td['center'][:,1], 'k--', lw=1.5, label='Center line')
    ax.plot(td['right'][:,0],  td['right'][:,1],  'r:',  lw=1.5, label='Right boundary')
    ax.plot(td['left'][:,0],   td['left'][:,1],   'b:',  lw=1.5, label='Left boundary')
    ax.plot(states[:,0], states[:,1], 'g', lw=2.2, label='MPC trajectory')

    for ob in obstacles or []:
        circ = Circle((float(ob["x"]), float(ob["y"])), float(ob["radius"]),
                      facecolor=(1.0, 0.4, 0.2, 0.35), edgecolor="r", lw=1.2, label="Obstacle")
        ax.add_patch(circ)

    if r_pts is not None and l_pts is not None:
        plot_boundary_markers(ax, r_pts, l_pts, put_legend=True)

    ax.set_title(title)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.grid(True); ax.axis('equal'); ax.legend(loc='upper right')
    plt.tight_layout(); plt.show()

# ----------------- the “missing” function -----------------

def plot_with_car_footprints(
    center_lane: np.ndarray,
    right_lane:  np.ndarray,
    left_lane:   np.ndarray,
    state_traj:  np.ndarray,          # (T,4): [x, y, psi, s]
    last_pred:   np.ndarray | None,   # (N+1,4) or None
    mpc=None,                          # pass your MPC instance to eval LUTs
    car_length: float = 0.58,
    car_width:  float = 0.30,
    anchor: str = "rear",              # "rear" or "center"
    draw_every: int = 1,               # draw every k-th step
    draw_corridor: bool = True,        # per-step corridor (left(s)↔right(s))
    draw_boundary_markers: bool = True,# little arrows showing normal
    draw_horizon_corridor: bool = True # corridor along last horizon
):
    """Track + driven path + last horizon + car rectangles + (optional) corridors/markers."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Track lines (static)
    ax.plot(center_lane[:, 0], center_lane[:, 1], "k--", lw=1.5, label="Center line")
    ax.plot(right_lane[:, 0],  right_lane[:, 1],  "r:",  lw=1.5, label="Right boundary")
    ax.plot(left_lane[:, 0],   left_lane[:, 1],   "b:",  lw=1.5, label="Left boundary")

    # Driven path
    ax.plot(state_traj[:, 0], state_traj[:, 1], color="tab:green", lw=2.0, label="MPC trajectory")

    # Last MPC horizon
    if last_pred is not None and last_pred.size:
        ax.plot(last_pred[:, 0], last_pred[:, 1], color="purple", lw=1.6, label="Last MPC horizon")

    # --- Per-step corridor + markers (needs mpc LUTs) ---
    if mpc is not None and draw_corridor:
        T = state_traj.shape[0]
        step_idx = range(0, T, max(1, int(draw_every)))

        qx, qy, qdx, qdy = [], [], [], []
        for i in step_idx:
            s = float(state_traj[i, 3])
            lx = float(mpc.left_lut_x(s));  ly = float(mpc.left_lut_y(s))
            rx = float(mpc.right_lut_x(s)); ry = float(mpc.right_lut_y(s))

            ax.plot([lx, rx], [ly, ry], color="0.5", alpha=0.35, lw=1.0)

            if draw_boundary_markers:
                tx, ty = (rx - lx), (ry - ly)
                nx, ny = -ty, tx
                seg_len = np.hypot(tx, ty)
                if seg_len > 1e-8:
                    nx /= seg_len; ny /= seg_len
                mx, my = (lx + rx) * 0.5, (ly + ry) * 0.5
                marker_len = 0.25 * seg_len
                qx.append(mx); qy.append(my); qdx.append(marker_len * nx); qdy.append(marker_len * ny)

        if draw_boundary_markers and len(qx):
            ax.quiver(qx, qy, qdx, qdy, angles='xy', scale_units='xy', scale=1.0,
                      width=0.003, color="0.3", alpha=0.6)

    # --- Corridor along last horizon (optional) ---
    if mpc is not None and draw_horizon_corridor and last_pred is not None and last_pred.size:
        for k in range(1, last_pred.shape[0]):  # skip k=0 if that's current state
            s = float(last_pred[k, 3])
            lx = float(mpc.left_lut_x(s));  ly = float(mpc.left_lut_y(s))
            rx = float(mpc.right_lut_x(s)); ry = float(mpc.right_lut_y(s))
            ax.plot([lx, rx], [ly, ry], color="purple", alpha=0.35, lw=1.2)

    # Car footprints (rectangles)
    cmap = mpl.cm.get_cmap("viridis")
    T = state_traj.shape[0]
    draw_idx = range(0, T, max(1, int(draw_every)))

    for i in draw_idx:
        x, y, psi, _ = state_traj[i, :]
        if anchor == "rear":
            cx = x + (car_length * 0.5) * np.cos(psi)
            cy = y + (car_length * 0.5) * np.sin(psi)
        else:
            cx, cy = x, y

        rect = Rectangle(
            (cx - car_length / 2.0, cy - car_width / 2.0),
            car_length, car_width,
            linewidth=0.6,
            edgecolor=cmap(i / max(1, T - 1)),
            facecolor=(*cmap(i / max(1, T - 1))[:3], 0.28),
        )
        trans = mpl.transforms.Affine2D().rotate_around(cx, cy, psi) + ax.transData
        rect.set_transform(trans)
        ax.add_patch(rect)

    # Styling
    ax.set_aspect("equal", "box")
    ax.grid(True, ls="--", alpha=0.35)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("MPC Simulated Trajectory (car footprints + corridor/markers)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# ----------------- playback animation -----------------

def animate_mpc_playback(
    center_lane: np.ndarray,
    right_lane: np.ndarray,
    left_lane: np.ndarray,
    states: np.ndarray,             # (T+1,4)
    ctrls: np.ndarray,              # (T,3)
    pred_log: list,                 # list of (N+1,4) or None
    mpc,                            # MPC instance (for left/right LUTs)
    obstacles_log: list | None = None,   # <— NEW: list[ list[ {x,y,radius} ] ]
    car_length: float = 0.58,
    car_width: float = 0.30,
    anchor: str = "rear",
    interval_ms: int = 50,
    save_path: str | None = None
):
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.collections import LineCollection
    from matplotlib.transforms import Affine2D

    T = states.shape[0] - 1
    fig, ax = plt.subplots(figsize=(11, 5.6))

    # static track
    ax.plot(center_lane[:,0], center_lane[:,1], "k--", lw=1.5, label="Center line")
    ax.plot(right_lane[:,0],  right_lane[:,1],  "r:",  lw=1.5, label="Right boundary")
    ax.plot(left_lane[:,0],   left_lane[:,1],   "b:",  lw=1.5, label="Left boundary")

    # driven path so far
    traj_line, = ax.plot([], [], color="tab:green", lw=2.2, label="MPC trajectory")

    # current corridor + horizon and horizon corridor
    corridor_line, = ax.plot([], [], color="0.4", lw=1.2, alpha=0.8, label="Corridor (current)")
    pred_line, = ax.plot([], [], color="purple", lw=1.6, alpha=0.9, label="Predicted horizon")
    horizon_corr_lc = LineCollection([], colors="purple", linewidths=1.2, alpha=0.35, label="Horizon corridor")
    ax.add_collection(horizon_corr_lc)

    # car rectangle
    car = Rectangle((0,0), car_length, car_width, facecolor=(0.1,0.6,1.0,0.35), edgecolor="tab:blue", lw=0.8)
    ax.add_patch(car)

    # moving obstacle patches (create from first frame)
    obs_patches = []
    if obstacles_log and len(obstacles_log) and len(obstacles_log[0]):
        for j, ob in enumerate(obstacles_log[0]):
            circ = Circle((float(ob["x"]), float(ob["y"])), float(ob.get("radius", 0.5)),
                          facecolor=(1.0, 0.4, 0.2, 0.35), edgecolor="r", lw=1.2)
            if j == 0:
                circ.set_label("Obstacle")
            ax.add_patch(circ)
            obs_patches.append(circ)

    # axes + legend
    ax.set_aspect("equal", "box")
    ax.grid(True, ls="--", alpha=0.35)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("MPC animation: car, corridor (current & horizon), horizon path")
    ax.legend(loc="upper right")

    # fixed limits
    margin = 1.0
    xmin = min(left_lane[:,0].min(), right_lane[:,0].min(), center_lane[:,0].min()) - margin
    xmax = max(left_lane[:,0].max(), right_lane[:,0].max(), center_lane[:,0].max()) + margin
    ymin = min(left_lane[:,1].min(), right_lane[:,1].min(), center_lane[:,1].min()) - margin
    ymax = max(left_lane[:,1].max(), right_lane[:,1].max(), center_lane[:,1].max()) + margin
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    info_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=10)

    def _place_car(i):
        x, y, psi, _ = states[i,:]
        if anchor == "rear":
            cx = x + (car_length * 0.5) * np.cos(psi)
            cy = y + (car_length * 0.5) * np.sin(psi)
        else:
            cx, cy = x, y
        car.set_xy((cx - car_length/2.0, cy - car_width/2.0))
        car.set_transform(Affine2D().rotate_around(cx, cy, psi) + ax.transData)

    def _update(frame):
        # driven path
        traj_line.set_data(states[:frame+1,0], states[:frame+1,1])

        # horizon path + corridor
        if 0 <= frame < len(pred_log) and pred_log[frame] is not None and pred_log[frame].size:
            pred = pred_log[frame]
            pred_line.set_data(pred[:,0], pred[:,1])
            segs = []
            for k in range(1, pred.shape[0]):
                s_k = float(pred[k, 3])
                lx = float(mpc.left_lut_x(s_k));  ly = float(mpc.left_lut_y(s_k))
                rx = float(mpc.right_lut_x(s_k)); ry = float(mpc.right_lut_y(s_k))
                segs.append([(lx, ly), (rx, ry)])
            horizon_corr_lc.set_segments(segs)
        else:
            pred_line.set_data([], [])
            horizon_corr_lc.set_segments([])

        # current corridor at robot state
        s = float(states[frame,3])
        lx = float(mpc.left_lut_x(s));  ly = float(mpc.left_lut_y(s))
        rx = float(mpc.right_lut_x(s)); ry = float(mpc.right_lut_y(s))
        corridor_line.set_data([lx, rx], [ly, ry])

        # move obstacle patches for this frame
        if obstacles_log and frame < len(obstacles_log) and len(obs_patches):
            obs_f = obstacles_log[frame]
            # assume constant count; update centers/radii
            for j, ob in enumerate(obs_f[:len(obs_patches)]):
                obs_patches[j].center = (float(ob["x"]), float(ob["y"]))
                obs_patches[j].radius = float(ob.get("radius", obs_patches[j].radius))

        _place_car(frame)

        v = float(ctrls[min(frame, ctrls.shape[0]-1), 0]) if ctrls.size else 0.0
        info_txt.set_text(f"step {frame+1}/{T+1}   v={v:.2f} m/s   s={s:.2f}")

        return (traj_line, pred_line, corridor_line, horizon_corr_lc, car, info_txt, *obs_patches)

    anim = animation.FuncAnimation(fig, _update, frames=T+1, interval=interval_ms, repeat=False)
    if save_path:
        ext = str(save_path).lower()
        if ext.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=max(1, int(1000/interval_ms)))
            anim.save(save_path, writer=writer, dpi=150)
        elif ext.endswith(".gif"):
            anim.save(save_path, dpi=150, writer="pillow")
    plt.show()
    return anim


# ----------------- dashboard animation -----------------

def animate_mpc_dashboard(
    td,
    states,                # (T,4): [x, y, psi, s]
    pred_hist,             # list of (N+1,4) or None per step; len = T
    dt,
    mpc,                   # MPC instance (for shifted corridor eval)
    obstacles_log=None,    # list[list[{x,y,radius}]] per frame (optional)
    v_hist=None,           # (T,) optional; if None we estimate from states
    v_ref=None,            # float or (T,) array; reference speed
    car_length=0.58,
    car_width=0.30,
    anchor="rear",         # "rear" -> (x,y) is rear axle; "center" -> rectangle center
    draw_every=1,
    fps=30,
    save_path=None         # e.g. "mpc_anim.mp4" or None for live
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle, Circle

    # ---------- prep time series ----------
    T = states.shape[0]
    t_axis = np.arange(T) * float(dt)

    # velocity: use provided or finite-difference from states
    if v_hist is None:
        if T < 2:
            v_hist = np.zeros(T)
        else:
            d = np.sqrt(np.diff(states[:,0])**2 + np.diff(states[:,1])**2)
            v_hist = np.r_[d[0]/dt, d/dt]
    else:
        v_hist = np.asarray(v_hist).reshape(-1)
        if v_hist.size != T:
            raise ValueError("v_hist length must equal states.shape[0]")

    # v_ref as array
    if v_ref is None:
        v_ref_arr = None
    else:
        if np.isscalar(v_ref):
            v_ref_arr = np.full(T, float(v_ref))
        else:
            v_ref_arr = np.asarray(v_ref).reshape(-1)
            if v_ref_arr.size != T:
                raise ValueError("v_ref array must have length states.shape[0]")

    # accelerations from kinematic signals
    psi = states[:,2]
    yaw_rate = np.gradient(psi, dt)
    a_lat = v_hist * yaw_rate
    a_long = np.gradient(v_hist, dt)

    # use one common limit so the friction circle is round
    a_lim = max(1e-6, np.percentile(np.abs(np.r_[a_lat, a_long]), 98))
    ay_n = a_lat / a_lim
    ax_n = a_long / a_lim

    # centerline LUTs
    c_x, c_y  = td['c_lut_x'], td['c_lut_y']
    c_dx, c_dy = td['c_lut_dx'], td['c_lut_dy']

    # ---------- figure & axes ----------
    plt.rcParams.update({"figure.autolayout": True})
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0])

    # Top: track view
    ax_tr = fig.add_subplot(gs[0, :])
    ax_tr.set_title("MPC dashboard: car, shifted corridor (current & horizon), moving obstacles")
    ax_tr.set_aspect("equal", "box")
    ax_tr.grid(True, ls="--", alpha=0.35)
    ax_tr.set_xlabel("X [m]"); ax_tr.set_ylabel("Y [m]")

    # Bottom-left: acceleration circle
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_acc.set_title("accelerations")
    ax_acc.set_xlabel(r"$a_y/a_{\max}$")
    ax_acc.set_ylabel(r"$a_x/a_{\max}$")
    ax_acc.grid(True, ls="--", alpha=0.35)

    # Bottom-middle: velocity
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_vel.set_title("velocity")
    ax_vel.set_xlabel("t [s]"); ax_vel.set_ylabel("v [m/s]")
    ax_vel.grid(True, ls="--", alpha=0.35)

    # Bottom-right: lateral deviation with shifted bounds
    ax_cte = fig.add_subplot(gs[1, 2])
    ax_cte.set_title("lateral deviation (shifted corridor bounds)")
    ax_cte.set_xlabel("t [s]"); ax_cte.set_ylabel("deviation [m]")
    ax_cte.grid(True, ls="--", alpha=0.35)

    # ---------- static track lines ----------
    ax_tr.plot(td['center'][:,0], td['center'][:,1], 'k--', lw=1.5, label="Center line")
    ax_tr.plot(td['right'][:,0],  td['right'][:,1],  'r:',  lw=1.5, label="Right boundary (base)")
    ax_tr.plot(td['left'][:,0],   td['left'][:,1],   'b:',  lw=1.5, label="Left boundary (base)")

    line_hist, = ax_tr.plot([], [], color="tab:green", lw=2.0, label="MPC trajectory")
    line_horz, = ax_tr.plot([], [], color="magenta", lw=1.8, label="Predicted horizon")

    # Corridor visuals (shifted)
    horizon_corr_segments = [ax_tr.plot([], [], color=(0.6,0.2,0.8,0.45), lw=1.5)[0] for _ in range(80)]
    corr_line, = ax_tr.plot([], [], color="0.2", lw=2.0, alpha=0.9, label="Corridor (current, shifted)")

    # Car rectangle
    car_rect = Rectangle((0,0), car_length, car_width, angle=0,
                         ec="tab:blue", fc=(0.3,0.5,1.0,0.6), lw=1.0)
    ax_tr.add_patch(car_rect)

    # Obstacles (moving) — create patches from first frame if available
    obs_patches = []
    if obstacles_log and len(obstacles_log) and len(obstacles_log[0]):
        for j, ob in enumerate(obstacles_log[0]):
            circ = Circle((float(ob["x"]), float(ob["y"])), float(ob.get("radius", 0.5)),
                          facecolor=(1.0, 0.4, 0.2, 0.35), edgecolor="r", lw=1.2)
            if j == 0:
                circ.set_label("Obstacle")
            ax_tr.add_patch(circ)
            obs_patches.append(circ)

    info_txt = ax_tr.text(0.02, 0.97, "", transform=ax_tr.transAxes,
                          ha="left", va="top",
                          bbox=dict(fc="white", ec="0.5", alpha=0.85, pad=2.5))
    ax_tr.legend(loc="upper right")
    offset_cm = 2.0

    # ---------- acceleration panel ----------
    theta = np.linspace(0, 2*np.pi, 256)
    ax_acc.plot(np.cos(theta), np.sin(theta), color="0.5", lw=1.0)  # unit circle
    ax_acc.set_aspect("equal", "box")
    ax_acc.set_xlim(-1.1, 1.1)
    ax_acc.set_ylim(-1.1, 1.1)

    sc = ax_acc.scatter([], [], c=[], cmap="viridis",
                        vmin=float(np.min(v_hist)), vmax=float(np.max(v_hist)),
                        s=14, edgecolors='none')
    cb = plt.colorbar(sc, ax=ax_acc, pad=0.02)
    cb.set_label("velocity [m/s]")

    # ---------- velocity panel ----------
    vel_ref_line, = ax_vel.plot([], [], color="tab:blue", lw=1.2, label="reference")
    vel_act_line, = ax_vel.plot([], [], color="tab:orange", lw=1.8, label="simulated")
    ax_vel.legend(loc="best")
    ax_vel.set_xlim(0, t_axis[-1] if T > 1 else 1.0)
    vmax = max(1.0, float(np.max(v_hist))*1.05)
    ax_vel.set_ylim(0, vmax)

    # ---------- CTE panel (shifted bounds) ----------
    # Pre-allocate (avoid append + duplicate-frame issues)
    draw_idx = np.arange(0, T, max(1, int(draw_every)))
    F = len(draw_idx)
    cte_hist_arr = np.full(F, np.nan)
    cte_low_arr  = np.full(F, np.nan)
    cte_high_arr = np.full(F, np.nan)

    cte_line,    = ax_cte.plot([], [], color="tab:blue", lw=1.8, label="CTE")
    cte_lo_line, = ax_cte.plot([], [], 'k--', lw=0.9, alpha=0.65, label="bounds (shifted)")
    cte_hi_line, = ax_cte.plot([], [], 'k--', lw=0.9, alpha=0.65)
    band = None
    ax_cte.legend(loc="best")

    # Track view limits (stable)
    margin = 1.0
    xmin = min(td['left'][:,0].min(), td['right'][:,0].min(), td['center'][:,0].min()) - margin
    xmax = max(td['left'][:,0].max(), td['right'][:,0].max(), td['center'][:,0].max()) + margin
    ymin = min(td['left'][:,1].min(), td['right'][:,1].min(), td['center'][:,1].min()) - margin
    ymax = max(td['left'][:,1].max(), td['right'][:,1].max(), td['center'][:,1].max()) + margin
    ax_tr.set_xlim(xmin, xmax); ax_tr.set_ylim(ymin, ymax)

    # CTE y-limits smoothing
    ax_cte.set_xlim(0, t_axis[draw_idx[-1]] if F > 1 else 1.0)
    ax_cte.set_ylim(-1.0, 1.0)
    cte_ylim = np.array([-1.0, 1.0], dtype=float)
    SMOOTH = 0.15

    def car_center_from_anchor(x, y, psi):
        if anchor == "rear":
            return (x + (car_length*0.5)*np.cos(psi),
                    y + (car_length*0.5)*np.sin(psi))
        return x, y

    def _lr_at_s(s):
        # prefer MPC LUTs if present, else use track td LUTs
        if (mpc is not None) and hasattr(mpc, 'left_lut_x'):
            lx = float(mpc.left_lut_x(s));
            ly = float(mpc.left_lut_y(s))
            rx = float(mpc.right_lut_x(s));
            ry = float(mpc.right_lut_y(s))
        else:
            lx = float(td['l_lut_x'](s));
            ly = float(td['l_lut_y'](s))
            rx = float(td['r_lut_x'](s));
            ry = float(td['r_lut_y'](s))
        return lx, ly, rx, ry

    def update(frame):
        nonlocal band, cte_ylim
        i = int(draw_idx[frame])

        # 0) apply shifted corridor for this frame (based on logged obstacles)
        can_shift = (mpc is not None) and hasattr(mpc, "apply_corridor_shift_for_obstacles") and hasattr(mpc,"clear_corridor_shifts")
        if obstacles_log and i < len(obstacles_log) and can_shift:
            mpc.apply_corridor_shift_for_obstacles(
                obstacles_log[i], s_window=8.0, inflate=0.7,
                far_side_nudge=0.10, Ns_field=400
            )
        elif can_shift:
            mpc.clear_corridor_shifts()

        # indices used up to this frame (decimated timeline)
        idx_used = draw_idx[:frame+1]
        t_draw = t_axis[idx_used]

        # 1) driven path
        line_hist.set_data(states[idx_used,0], states[idx_used,1])

        # 2) car pose at this frame
        #x, y, psi_i, s = states[i]
        x, y, psi_i, s = states[i, :4]
        cx, cy = car_center_from_anchor(x, y, psi_i)
        car_rect.set_xy((cx - car_length/2.0, cy - car_width/2.0))
        car_rect.angle = np.degrees(psi_i)

        # 3) current corridor (SHIFTED)
        lx, ly, rx, ry = _lr_at_s(float(s))
        corr_line.set_data([lx, rx], [ly, ry])

        # 4) predicted horizon + its (SHIFTED) corridor
        ph = pred_hist[i] if (pred_hist is not None and len(pred_hist) > i) else None
        if ph is not None and ph.size:
            line_horz.set_data(ph[:,0], ph[:,1])
            K = min(len(horizon_corr_segments), ph.shape[0]-1)
            for k in range(K):
                s_k = float(ph[k+1, 3])
                lxk, lyk, rxk, ryk = _lr_at_s(s_k)
                horizon_corr_segments[k].set_data([lxk, rxk], [lyk, ryk])
                horizon_corr_segments[k].set_visible(True)
            for k in range(K, len(horizon_corr_segments)):
                horizon_corr_segments[k].set_visible(False)
        else:
            line_horz.set_data([], [])
            for seg in horizon_corr_segments:
                seg.set_visible(False)

        # 5) moving obstacles (update centers)
        if obstacles_log and i < len(obstacles_log) and len(obs_patches):
            obs_f = obstacles_log[i]
            for j, ob in enumerate(obs_f[:len(obs_patches)]):
                obs_patches[j].center = (float(ob["x"]), float(ob["y"]))
                obs_patches[j].radius = float(ob.get("radius", obs_patches[j].radius))

        # 6) acceleration scatter (normalized → round circle) on decimated timeline
        sc.set_offsets(np.column_stack([ay_n[idx_used], ax_n[idx_used]]))
        sc.set_array(v_hist[idx_used])

        # 7) velocity (reference + simulated) on decimated timeline
        if v_ref_arr is None:
            vel_ref_line.set_data([], [])
        else:
            vel_ref_line.set_data(t_draw, v_ref_arr[idx_used])
        vel_act_line.set_data(t_draw, v_hist[idx_used])
        xpair_vel = [0.0, ax_vel.get_xlim()[1]]
        vel_min_line.set_data(xpair_vel, [float(lim["vx_min"]), float(lim["vx_min"])])
        vel_max_line.set_data(xpair_vel, [float(lim["vx_max"]), float(lim["vx_max"])])

        # 8) CTE + SHIFTED bounds + band; smooth auto-scale
        cx_s, cy_s = float(c_x(s)), float(c_y(s))
        dx_s, dy_s = float(c_dx(s)), float(c_dy(s))
        th = np.arctan2(dy_s, dx_s)

        cte_i = np.sin(th)*(x - cx_s) - np.cos(th)*(y - cy_s)
        cte_l = np.sin(th)*(lx - cx_s) - np.cos(th)*(ly - cy_s)
        cte_r = np.sin(th)*(rx - cx_s) - np.cos(th)*(ry - cy_s)
        lo_i, hi_i = (min(cte_l, cte_r), max(cte_l, cte_r))

        # store at 'frame' index (no appends)
        cte_hist_arr[frame] = cte_i
        cte_low_arr[frame]  = lo_i
        cte_high_arr[frame] = hi_i

        # draw with matching lengths
        cte_line.set_data(t_draw,   cte_hist_arr[:frame+1])
        cte_lo_line.set_data(t_draw, cte_low_arr[:frame+1])
        cte_hi_line.set_data(t_draw, cte_high_arr[:frame+1])
        if band is not None:
            band.remove()
        band = ax_cte.fill_between(t_draw,
                                   cte_low_arr[:frame+1],
                                   cte_high_arr[:frame+1],
                                   color='0.85', alpha=0.45)

        target = max(0.5, 1.1*np.nanmax(np.abs(
            np.r_[cte_hist_arr[:frame+1], cte_low_arr[:frame+1], cte_high_arr[:frame+1]]
        )))
        cte_ylim = (1.0 - SMOOTH) * cte_ylim + SMOOTH * np.array([-target, target])
        ax_cte.set_ylim(cte_ylim[0], cte_ylim[1])

        # 9) HUD
        info_txt.set_text(f"step {i+1}/{T}   v={v_hist[i]:.2f} m/s   s={states[i,3]:.2f}")

        return (line_hist, car_rect, corr_line, line_horz, sc,
                vel_ref_line, vel_act_line, cte_line, cte_lo_line, cte_hi_line, info_txt, *obs_patches)

    anim = FuncAnimation(
        fig, update, frames=len(draw_idx),
        interval=1000.0/fps, blit=False, repeat=False
    )

    if save_path:
        print(f"Saving animation to: {save_path}")
        anim.save(save_path, fps=fps, dpi=140)
    else:
        plt.show()

    # restore base corridor after anim (no side effects)

    if (mpc is not None) and hasattr(mpc, "clear_corridor_shifts"):
        mpc.clear_corridor_shifts()
    return anim, fig

def animate_mpc_dashboard_diagnosis(
    td,
    states,                # (T,4): [x, y, psi, s]
    pred_hist,             # list of (N+1,4) or None per step; len = T
    dt,
    mpc,                   # MPC instance (for shifted corridor eval)
    obstacles_log=None,    # list[list[{x,y,radius}]] per frame (optional)
    v_hist=None,           # (T,) optional; if None we estimate from states
    v_ref=None,            # float or (T,) array; reference speed
    car_length=0.58,
    car_width=0.30,
    anchor="rear",         # "rear" -> (x,y) is rear axle; "center" -> rectangle center
    draw_every=1,
    fps=30,
    save_path=None,        # e.g. "mpc_anim.mp4" or None for live

    # ------ optional diagnostics ------
    ctrls=None,            # (T,3) or (T-1,3) -> for steering θ plot (deg)
    cost_hist=None,        # (T,) MPC objective per solve
    solve_time=None,       # (T,) solver time per step [s]
    lap_s0=0.0,            # start/finish line location along s
    show_start_line=True,

    # heading plot selection: 'all', 'psi', 'phi', 'beta', or list like ['psi','beta']
    head_show='all',

    # NEW: predicted velocity over the horizon per step (if None, it will be derived from pred_hist)
    u_pred_hist=None,      # list of arrays (len N or N+1) with predicted speed [m/s] per step
    limits=None,           # optional dict with constraint limits for visualization
):
    """
    Dashboard with track + diagnostics.

    Changes:
      • Legend sits outside (right of track) and the info text panel is under that legend.
      • Acceleration plot spans two rows.
      • Velocity panel has a third curve (magenta) showing horizon prediction u_pred.
        If u_pred_hist is None, it is derived from pred_hist geometry.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle, Circle

    # ---------- helpers ----------
    def _wrap_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def _break_wrap(deg_series, jump=180.0):
        s = np.asarray(deg_series, dtype=float).copy()
        if s.size <= 1:
            return s
        d = np.diff(s)
        jumps = np.where(np.abs(d) > jump)[0]
        if jumps.size == 0:
            return s
        out = []
        last = 0
        for j in jumps:
            out.extend(s[last:j+1].tolist())
            out.append(np.nan)
            last = j+1
        out.extend(s[last:].tolist())
        return np.array(out, dtype=float)

    def _lr_at_s(s):
        # prefer MPC LUTs if present, else use track td LUTs
        if (mpc is not None) and hasattr(mpc, 'left_lut_x'):
            lx = float(mpc.left_lut_x(s)); ly = float(mpc.left_lut_y(s))
            rx = float(mpc.right_lut_x(s)); ry = float(mpc.right_lut_y(s))
        else:
            lx = float(td['l_lut_x'](s)); ly = float(td['l_lut_y'](s))
            rx = float(td['r_lut_x'](s)); ry = float(td['r_lut_y'](s))
        return lx, ly, rx, ry

    def _speed_from_pred_states(ph):
        """Derive horizon speed from predicted states (x,y,psi,s)."""
        if ph is None or ph.size == 0 or ph.shape[0] < 2:
            return np.array([])
        d = np.sqrt(np.diff(ph[:,0])**2 + np.diff(ph[:,1])**2)
        return d / float(dt)  # length N (between nodes)

    # ---------- prep time series ----------
    T = int(states.shape[0])
    t_axis = np.arange(T) * float(dt)

    # velocity
    if v_hist is None:
        if T < 2:
            v_hist = np.zeros(T)
        else:
            d = np.sqrt(np.diff(states[:,0])**2 + np.diff(states[:,1])**2)
            v_hist = np.r_[d[0]/dt, d/dt]
    else:
        v_hist = np.asarray(v_hist).reshape(-1)
        if v_hist.size != T:
            raise ValueError("v_hist length must equal states.shape[0]")

    # v_ref as array
    if v_ref is None:
        v_ref_arr = None
    else:
        v_ref_arr = (np.full(T, float(v_ref)) if np.isscalar(v_ref)
                     else np.asarray(v_ref).reshape(-1))
        if v_ref_arr.size != T:
            raise ValueError("v_ref array must have length states.shape[0]")

    # steering history (→ degrees)
    theta_hist = None
    th_max_deg = None
    D_hist = None
    vs_hist = None
    if ctrls is not None:
        C = np.asarray(ctrls)
        if C.ndim < 2 or C.shape[1] < 2:
            raise ValueError("ctrls must have shape (T or T-1, >=2).")
        if C.shape[0] == T-1:
            theta_hist = np.degrees(np.r_[C[0,1], C[:,1]])
        elif C.shape[0] == T:
            theta_hist = np.degrees(C[:,1])
        else:
            theta_hist = np.degrees(C[:T,1] if C.shape[0] >= T
                                    else np.r_[C[0,1], C[:,1], np.repeat(C[-1,1], T-1-C.shape[0])])
        if hasattr(mpc, "theta_max"):
            th_max_deg = float(np.degrees(mpc.theta_max))

        def _align_ctrl_col(arr, col, target_len):
            src = np.asarray(arr[:, col]).reshape(-1)
            if src.size == target_len - 1:
                return np.r_[src[0], src]
            if src.size >= target_len:
                return src[:target_len]
            if src.size == 0:
                return np.zeros(target_len, dtype=float)
            return np.r_[src, np.repeat(src[-1], target_len - src.size)]

        if C.shape[1] >= 1:
            D_hist = _align_ctrl_col(C, 0, T)
        if C.shape[1] >= 3:
            vs_hist = _align_ctrl_col(C, 2, T)

    lim = {} if limits is None else dict(limits)

    def _lim_default(key, value):
        if key not in lim or lim[key] is None:
            lim[key] = value

    _lim_default("theta_max", float(getattr(mpc, "theta_max", 0.35)))
    _lim_default("D_min", -0.1)
    _lim_default("D_max", 1.0)
    _lim_default("vs_min", 0.0)
    _lim_default("vs_max", float(np.nanmax(v_hist)) if v_hist.size else 3.5)
    _lim_default("vx_min", 0.0)
    _lim_default("vx_max", float(np.nanmax(v_hist)) if v_hist.size else 3.5)

    # accelerations
    psi = states[:,2]
    yaw_rate = np.gradient(psi, dt)
    a_lat = v_hist * yaw_rate
    a_long = np.gradient(v_hist, dt)
    a_lim = max(1e-6, np.percentile(np.abs(np.r_[a_lat, a_long]), 98))
    ay_n = a_lat / a_lim
    ax_n = a_long / a_lim

    # centerline LUTs
    c_x, c_y  = td['c_lut_x'], td['c_lut_y']
    c_dx, c_dy = td['c_lut_dx'], td['c_lut_dy']

    # headings & errors in degrees
    phi = np.arctan2(
        np.array([float(c_dy(float(s))) for s in states[:,3]]),
        np.array([float(c_dx(float(s))) for s in states[:,3]]))
    beta = _wrap_pi(psi - phi)
    psi_deg  = np.degrees(psi)
    phi_deg  = np.degrees(phi)
    beta_deg = np.degrees(beta)

    # ---------- lap timing ----------
    L = float(td['s_total'])
    s_log = states[:,3].astype(float)
    s_unwrap = np.empty_like(s_log)
    offset = 0.0
    s_unwrap[0] = s_log[0]
    for i in range(1, T):
        ds = s_log[i] - s_log[i-1]
        if ds < -0.5*L: offset += L
        elif ds > 0.5*L: offset -= L
        s_unwrap[i] = s_log[i] + offset

    def compute_lap_crossings(t, s_unw, s0, L):
        t_cross = []
        n_min = int(np.floor((s_unw[0] - s0)/L))
        n_max = int(np.floor((s_unw[-1] - s0)/L))
        for n in range(n_min+1, n_max+1):
            target = s0 + n*L
            idx = np.searchsorted(s_unw, target)
            if 0 < idx < len(s_unw):
                i0 = idx-1; i1 = idx
                s0_i, s1_i = s_unw[i0], s_unw[i1]
                alpha = 0.0 if s1_i == s0_i else (target - s0_i) / (s1_i - s0_i)
                t_cross.append(t[i0] + alpha*(t[i1] - t[i0]))
        return np.array(t_cross, dtype=float)

    t_cross = compute_lap_crossings(t_axis, s_unwrap, float(lap_s0), L)
    lap_times = np.diff(t_cross) if t_cross.size >= 2 else np.array([])
    last5 = lap_times[-5:] if lap_times.size else np.array([])

    # ---------- figure & axes ----------
    plt.rcParams.update({"figure.autolayout": True})
    fig = plt.figure(figsize=(15.5, 9.3))
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[2.3, 1.0, 1.0],
        width_ratios=[1.1, 1.0, 1.0, 0.9]
    )

    # Top: track spans columns 0..2
    ax_tr = fig.add_subplot(gs[0, 0:3])
    ax_tr.set_title("MPC dashboard: car, shifted corridor (current & horizon), moving obstacles")
    ax_tr.set_aspect("equal", "box")
    ax_tr.grid(True, ls="--", alpha=0.35)
    ax_tr.set_xlabel("X [m]"); ax_tr.set_ylabel("Y [m]")

    # Top-right: drivetrain input D
    ax_D = fig.add_subplot(gs[0, 3])

    # Row 2 (index 1)
    ax_acc  = fig.add_subplot(gs[1:, 0])  # spans two rows
    ax_vel  = fig.add_subplot(gs[1, 1])
    ax_cte  = fig.add_subplot(gs[1, 2])
    ax_cost = fig.add_subplot(gs[1, 3])

    # Row 3 (index 2)
    ax_head = fig.add_subplot(gs[2, 1])
    ax_steer= fig.add_subplot(gs[2, 2])
    ax_solv = fig.add_subplot(gs[2, 3])

    # ---------- static track lines ----------
    ax_tr.plot(td['center'][:,0], td['center'][:,1], 'k--', lw=1.5, label="Center line")
    ax_tr.plot(td['right'][:,0],  td['right'][:,1],  'r:',  lw=1.5, label="Right boundary (base)")
    ax_tr.plot(td['left'][:,0],   td['left'][:,1],   'b:',  lw=1.5, label="Left boundary (base)")

    # Start/finish line
    if show_start_line:
        lx0, ly0, rx0, ry0 = _lr_at_s(float(lap_s0))
        ax_tr.plot([lx0, rx0], [ly0, ry0], color="0.1", lw=3.0, solid_capstyle="butt", label="Start/Finish")

    line_hist, = ax_tr.plot([], [], color="tab:green", lw=2.0, label="MPC trajectory")
    line_horz, = ax_tr.plot([], [], color="magenta", lw=1.8, label="Predicted horizon")
    horizon_corr_lc = LineCollection([], colors=(0.6, 0.2, 0.8, 0.45), linewidths=1.5)
    ax_tr.add_collection(horizon_corr_lc)
    corr_line, = ax_tr.plot([], [], color="0.2", lw=2.0, alpha=0.9, label="Corridor (current, shifted)")

    # Car rectangle
    car_rect = Rectangle((0,0), car_length, car_width, angle=0,
                         ec="tab:blue", fc=(0.3,0.5,1.0,0.6), lw=1.0)
    ax_tr.add_patch(car_rect)

    # Obstacles
    obs_patches = []
    if obstacles_log and len(obstacles_log) and len(obstacles_log[0]):
        for j, ob in enumerate(obstacles_log[0]):
            circ = Circle((float(ob["x"]), float(ob["y"])), float(ob.get("radius", 0.5)),
                          facecolor=(1.0, 0.4, 0.2, 0.35), edgecolor="r", lw=1.2)
            if j == 0:
                circ.set_label("Obstacle")
            ax_tr.add_patch(circ)
            obs_patches.append(circ)

    # Legend outside (right of track)
    leg = ax_tr.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)

    # Info text under the legend (outside the axes)
    def _format_laptimes(now_t):
        if t_cross.size == 0 or now_t < t_cross[0]:
            return "Last lap: –\nBest lap: –\nCurrent: –\nLast 5: –"
        idx = np.searchsorted(t_cross, now_t, side='right')
        best = np.nanmin(np.diff(t_cross[:idx])) if idx >= 2 else np.nan
        last = (t_cross[idx-1] - t_cross[idx-2]) if idx >= 2 else np.nan
        cur  = now_t - (t_cross[idx-1] if idx >= 1 else t_cross[0])
        def fmt(x): return "-" if (not np.isfinite(x)) else f"{x:5.2f} s"
        last5_txt = ""
        if lap_times.size:
            L5 = lap_times[:idx-1][-4:]
            last5_txt = ", ".join(f"{v:.2f}" for v in L5) if L5.size else "-"
        else:
            last5_txt = "-"
        return (f"Last lap:  {fmt(last)}\n"
                f"Best lap:  {fmt(best)}\n"
                f"Current:   {fmt(cur)}\n"
                f"Last 4:    {last5_txt}\n")

    info_txt = ax_tr.text(
        1.02, 0.42, _format_laptimes(0.0),
        transform=ax_tr.transAxes, ha="left", va="top",
        fontsize=10,
        bbox=dict(fc="white", ec="0.6", alpha=0.95, pad=6.0)
    )

    # ---------- drivetrain input D ----------
    ax_D.set_title("drivetrain input D")
    ax_D.set_xlabel("t [s]")
    ax_D.set_ylabel("D")
    ax_D.grid(True, ls="--", alpha=0.35)
    D_line, = ax_D.plot([], [], lw=1.8, color="tab:purple", label="D")
    D_min_line, = ax_D.plot([], [], lw=1.0, ls='--', color="0.5", label="D_min")
    D_max_line, = ax_D.plot([], [], lw=1.0, ls='--', color="0.5", label="D_max")
    ax_D.legend(loc="upper right", fontsize=8)
    ax_D.set_xlim(0.0, t_axis[-1] if T > 1 else 1.0)
    D_lim_lo = float(lim["D_min"])
    D_lim_hi = float(lim["D_max"])
    D_span = max(0.2, D_lim_hi - D_lim_lo)
    ax_D.set_ylim(D_lim_lo - 0.25 * D_span, D_lim_hi + 0.25 * D_span)

    D_box = ax_D.text(
        0.98, 0.02, "", transform=ax_D.transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(fc="white", ec="0.6", alpha=0.9, pad=4.0)
    )

    # ---------- acceleration panel (spans two rows) ----------
    theta_circ = np.linspace(0, 2*np.pi, 256)
    ax_acc.plot(np.cos(theta_circ), np.sin(theta_circ), color="0.5", lw=1.0)  # unit circle
    ax_acc.set_aspect("equal", "box")
    ax_acc.set_xlim(-1.1, 1.1)
    ax_acc.set_ylim(-1.1, 1.1)
    ax_acc.set_title("accelerations")
    ax_acc.set_xlabel(r"$a_y/a_{\max}$")
    ax_acc.set_ylabel(r"$a_x/a_{\max}$")
    ax_acc.grid(True, ls="--", alpha=0.35)

    sc = ax_acc.scatter([], [], c=[], cmap="viridis",
                        vmin=float(np.min(v_hist)), vmax=float(np.max(v_hist)),
                        s=14, edgecolors='none')
    cb = plt.colorbar(sc, ax=ax_acc, pad=0.02)
    cb.set_label("velocity [m/s]")

    # ---------- velocity panel ----------
    ax_vel.set_title("velocity")
    ax_vel.set_xlabel("t [s]"); ax_vel.set_ylabel("v [m/s]")
    ax_vel.grid(True, ls="--", alpha=0.35)
    vel_ref_line,  = ax_vel.plot([], [], color="tab:blue",   lw=1.2, label="reference")
    vel_act_line,  = ax_vel.plot([], [], color="tab:orange", lw=1.8, label="simulated")
    vel_pred_line, = ax_vel.plot([], [], color="magenta", linestyle = "dotted",   lw=1.5, alpha=0.85, label="predicted horizon")  # NEW
    vel_min_line = ax_vel.plot([], [], lw=1.0, ls='--', color="0.5", label="v_min")[0]
    vel_max_line = ax_vel.plot([], [], lw=1.0, ls='--', color="0.5", label="v_max")[0]
    ax_vel.legend(loc="best")
    ax_vel.set_xlim(0, t_axis[-1] if T > 1 else 1.0)
    vmax0 = max(1.0, float(np.max(v_hist))*1.05, float(lim["vx_max"]) * 1.05)
    vmin0 = min(0.0, float(lim["vx_min"]) * 1.05)
    ax_vel.set_ylim(vmin0, vmax0)

    # ---------- CTE panel ----------
    draw_idx = np.arange(0, T, max(1, int(draw_every)))
    F = len(draw_idx)
    cte_hist_arr = np.full(F, np.nan)
    cte_low_arr  = np.full(F, np.nan)
    cte_high_arr = np.full(F, np.nan)

    ax_cte.set_title("lateral deviation (shifted corridor bounds)")
    ax_cte.set_xlabel("t [s]"); ax_cte.set_ylabel("deviation [m]")
    ax_cte.grid(True, ls="--", alpha=0.35)
    cte_line,    = ax_cte.plot([], [], color="tab:blue", lw=1.8, label="CTE")
    cte_lo_line, = ax_cte.plot([], [], 'k--', lw=0.9, alpha=0.65, label="bounds (shifted)")
    cte_hi_line, = ax_cte.plot([], [], 'k--', lw=0.9, alpha=0.65)
    band = None
    ax_cte.legend(loc="best")
    ax_cte.set_xlim(0, t_axis[draw_idx[-1]] if F > 1 else 1.0)
    ax_cte.set_ylim(-1.0, 1.0)
    cte_ylim = np.array([-1.0, 1.0], dtype=float)
    CTE_SMOOTH = 0.15

    # ---------- cost panel ----------
    ax_cost.set_title("MPC cost")
    ax_cost.set_xlabel("t [s]"); ax_cost.set_ylabel("cost")
    ax_cost.grid(True, ls="--", alpha=0.35)
    cost_line, = ax_cost.plot([], [], lw=1.8, color="tab:purple", label="objective")
    if cost_hist is not None:
        cost_hist = np.asarray(cost_hist).reshape(-1)
        if cost_hist.size != T:
            cost_hist = (np.r_[cost_hist[0], cost_hist] if cost_hist.size == T-1
                         else np.pad(cost_hist[:T], (0, max(0, T - cost_hist.size)), mode='edge'))
        ax_cost.set_xlim(0, t_axis[-1] if T > 1 else 1.0)

    # ---------- heading (degrees) ----------
    ax_head.set_title("heading ψ, tangent φ, and β = wrap(ψ-φ)  [deg]")
    ax_head.set_xlabel("t [s]"); ax_head.set_ylabel("angle [deg]")
    ax_head.grid(True, ls="--", alpha=0.35)

    show_set = {'psi','phi','beta'} if head_show == 'all' else set([head_show] if isinstance(head_show, str) else head_show)
    head_psi_line = ax_head.plot([], [], lw=1.4, color="tab:green",  label="ψ (heading)")[0] if 'psi' in show_set else None
    head_phi_line = ax_head.plot([], [], lw=1.0, color="tab:gray",   label="φ (tangent)")[0] if 'phi' in show_set else None
    head_beta_line= ax_head.plot([], [], lw=1.6, color="tab:red",    label="β (heading error)")[0] if 'beta' in show_set else None
    if show_set:
        ax_head.legend(loc="best")
    ax_head.set_xlim(0, t_axis[-1] if T > 1 else 1.0)
    head_ylim = np.array([-30.0, 30.0], dtype=float)
    HEAD_SMOOTH = 0.20

    # ---------- steering (degrees) ----------
    T_total = float(t_axis[draw_idx[-1]]) if len(draw_idx) > 1 else float(dt)
    ax_steer.set_title("steering input θ [deg]")
    ax_steer.set_xlabel("t [s]"); ax_steer.set_ylabel("θ [deg]")
    ax_steer.grid(True, ls="--", alpha=0.35)
    steer_line, = ax_steer.plot([], [], lw=1.8, color="tab:olive")
    ax_steer.set_xlim(0.0, T_total)

    if np.isfinite(float(lim["theta_max"])):
        th_max_deg = float(np.degrees(float(lim["theta_max"])))

    if th_max_deg is not None:
        steer_min_line, = ax_steer.plot([], [], lw=1.0, ls='--', color="0.5")
        steer_max_line, = ax_steer.plot([], [], lw=1.0, ls='--', color="0.5")
    else:
        steer_min_line = steer_max_line = None

    steer_box = ax_steer.text(
        0.98, 0.02, "", transform=ax_steer.transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(fc="white", ec="0.6", alpha=0.9, pad=4.0)
    )

    # ---------- solver time ----------
    ax_solv.set_title("solver time")
    ax_solv.set_xlabel("t [s]"); ax_solv.set_ylabel("time [ms]")
    ax_solv.grid(True, ls="--", alpha=0.35)
    solv_line, = ax_solv.plot([], [], lw=1.8, color="tab:cyan")
    ax_solv.set_xlim(0.0, T_total)

    solv_box = ax_solv.text(
        0.98, 0.02, "", transform=ax_solv.transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(fc="white", ec="0.6", alpha=0.9, pad=4.0)
    )

    # ---------- track view limits ----------
    margin = 1.0
    xmin = min(td['left'][:,0].min(), td['right'][:,0].min(), td['center'][:,0].min()) - margin
    xmax = max(td['left'][:,0].max(), td['right'][:,0].max(), td['center'][:,0].max()) + margin
    ymin = min(td['left'][:,1].min(), td['right'][:,1].min(), td['center'][:,1].min()) - margin
    ymax = max(td['left'][:,1].max(), td['right'][:,1].max(), td['center'][:,1].max()) + margin
    ax_tr.set_xlim(xmin, xmax); ax_tr.set_ylim(ymin, ymax)

    def _xy_with_wrap_breaks(x, y, jump=180.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.size <= 1:
            return x, y
        dy = np.diff(y)
        xb = [x[0]]; yb = [y[0]]
        for k in range(1, y.size):
            if abs(dy[k - 1]) > jump:
                xb.append(np.nan); yb.append(np.nan)
            xb.append(x[k]); yb.append(y[k])
        return np.asarray(xb), np.asarray(yb)

    def car_center_from_anchor(x, y, psi):
        if anchor == "rear":
            return (x + (car_length*0.5)*np.cos(psi),
                    y + (car_length*0.5)*np.sin(psi))
        return x, y

    # ---------- animator ----------
    def update(frame):
        nonlocal band, cte_ylim, head_ylim
        i = int(draw_idx[frame])

        # corridor shift for this frame
        can_shift = (mpc is not None) and hasattr(mpc, "apply_corridor_shift_for_obstacles") and hasattr(mpc, "clear_corridor_shifts")
        if obstacles_log and i < len(obstacles_log) and can_shift:
            mpc.apply_corridor_shift_for_obstacles(obstacles_log[i], s_window=8.0, inflate=0.7,
                                                   far_side_nudge=0.10, Ns_field=400)
        elif can_shift:
            mpc.clear_corridor_shifts()

        idx_used = draw_idx[:frame + 1]
        t_draw = t_axis[idx_used]

        # path + car
        line_hist.set_data(states[idx_used, 0], states[idx_used, 1])
        x, y, psi_i, s = states[i]
        cx, cy = car_center_from_anchor(x, y, psi_i)
        car_rect.set_xy((cx - car_length / 2.0, cy - car_width / 2.0))
        car_rect.angle = np.degrees(psi_i)

        # current corridor
        lx, ly, rx, ry = _lr_at_s(float(s))
        corr_line.set_data([lx, rx], [ly, ry])

        # predicted horizon + corridor
        ph = pred_hist[i] if (pred_hist is not None and len(pred_hist) > i) else None
        if ph is not None and ph.size:
            line_horz.set_data(ph[:, 0], ph[:, 1])
            segs = []
            for k in range(ph.shape[0] - 1):
                s_k = float(ph[k + 1, 3])
                lxk, lyk, rxk, ryk = _lr_at_s(s_k)
                segs.append([(lxk, lyk), (rxk, ryk)])
            horizon_corr_lc.set_segments(segs)
        else:
            line_horz.set_data([], [])
            horizon_corr_lc.set_segments([])

        # obstacles move
        if obstacles_log and i < len(obstacles_log) and len(obs_patches):
            obs_f = obstacles_log[i]
            for j, ob in enumerate(obs_f[:len(obs_patches)]):
                obs_patches[j].center = (float(ob["x"]), float(ob["y"]))
                obs_patches[j].radius = float(ob.get("radius", obs_patches[j].radius))

        # accel scatter
        sc.set_offsets(np.column_stack([ay_n[idx_used], ax_n[idx_used]]))
        sc.set_array(v_hist[idx_used])

        # ---------- velocity (act + ref) ----------
        if v_ref_arr is None:
            vel_ref_line.set_data([], [])
        else:
            vel_ref_line.set_data(t_draw, v_ref_arr[idx_used])
        vel_act_line.set_data(t_draw, v_hist[idx_used])

        # ---------- NEW: predicted horizon velocity ----------
        u_pred_i = None
        if u_pred_hist is not None and i < len(u_pred_hist) and u_pred_hist[i] is not None:
            u_pred_i = np.asarray(u_pred_hist[i]).reshape(-1)
        else:
            if ph is not None:
                u_pred_i = _speed_from_pred_states(ph)  # length N

        if u_pred_i is not None and u_pred_i.size:
            t0 = t_axis[i]
            Np = u_pred_i.size
            t_future = t0 + dt * (np.arange(1, Np+1))      # t+dt ... t+N*dt
            t_pred_plot = np.r_[t0, t_future]              # prepend current time
            v_pred_plot = np.r_[v_hist[i], u_pred_i]       # continuity
            vel_pred_line.set_data(t_pred_plot, v_pred_plot)

            # keep limits generous
            vmax_now = max(ax_vel.get_ylim()[1], float(np.nanmax(v_pred_plot))*1.05, float(np.nanmax(v_hist))*1.05, float(lim["vx_max"]) * 1.05, 1.0)
            vmin_now = min(ax_vel.get_ylim()[0], float(lim["vx_min"]) * 1.05, 0.0)
            ax_vel.set_ylim(vmin_now, vmax_now)
            x_max_now = ax_vel.get_xlim()[1]
            if t_pred_plot[-1] > x_max_now:
                ax_vel.set_xlim(0.0, t_pred_plot[-1] * 1.02)
        else:
            vel_pred_line.set_data([], [])

        # CTE + bounds
        cx_s, cy_s = float(c_x(s)), float(c_y(s))
        dx_s, dy_s = float(c_dx(s)), float(c_dy(s))
        th = np.arctan2(dy_s, dx_s)
        cte_i = np.sin(th) * (x - cx_s) - np.cos(th) * (y - cy_s)
        cte_l = np.sin(th) * (lx - cx_s) - np.cos(th) * (ly - cy_s)
        cte_r = np.sin(th) * (rx - cx_s) - np.cos(th) * (ry - cy_s)
        lo_i, hi_i = (min(cte_l, cte_r), max(cte_l, cte_r))
        cte_hist_arr[frame] = cte_i; cte_low_arr[frame] = lo_i; cte_high_arr[frame] = hi_i
        cte_line.set_data(t_draw, cte_hist_arr[:frame + 1])
        cte_lo_line.set_data(t_draw, cte_low_arr[:frame + 1])
        cte_hi_line.set_data(t_draw, cte_high_arr[:frame + 1])
        if band is not None: band.remove()
        band = ax_cte.fill_between(t_draw, cte_low_arr[:frame + 1], cte_high_arr[:frame + 1],
                                   color='0.85', alpha=0.45)
        target = max(0.5, 1.1 * np.nanmax(np.abs(
            np.r_[cte_hist_arr[:frame + 1], cte_low_arr[:frame + 1], cte_high_arr[:frame + 1]]
        )))
        cte_ylim = (1.0 - CTE_SMOOTH) * cte_ylim + CTE_SMOOTH * np.array([-target, target])
        ax_cte.set_ylim(cte_ylim[0], cte_ylim[1])

        # drivetrain input D with hard limits
        xpair_D = [0.0, T_total]
        D_min_line.set_data(xpair_D, [float(lim["D_min"]), float(lim["D_min"])])
        D_max_line.set_data(xpair_D, [float(lim["D_max"]), float(lim["D_max"])])
        if D_hist is not None:
            D_used = D_hist[idx_used]
            D_line.set_data(t_draw, D_used)
            d_avg = np.nanmean(D_used); d_min = np.nanmin(D_used); d_max = np.nanmax(D_used)
            D_box.set_text(f"avg: {d_avg:4.2f}\nmin: {d_min:4.2f}\nmax: {d_max:4.2f}")
        else:
            D_line.set_data([], [])
            D_box.set_text("D input unavailable")

        # cost panel
        if cost_hist is not None:
            ch = np.asarray(cost_hist)
            iu = idx_used[idx_used < ch.size]
            td_ = t_axis[iu]
            cost_line.set_data(td_, ch[iu])
            avg_abs = max(1e-6, float(np.nanmean(np.abs(ch[:i + 1]))))
            ax_cost.set_ylim(-2.5 * avg_abs, 2.5 * avg_abs)
            ax_cost.set_xlim(0, t_axis[draw_idx[-1]])

        # headings (deg)
        series_used = []
        if head_psi_line is not None:
            xw, yw = _xy_with_wrap_breaks(t_draw, psi_deg[idx_used], jump=180.0)
            head_psi_line.set_data(xw, yw); series_used.append(yw)
        if head_phi_line is not None:
            xw, yw = _xy_with_wrap_breaks(t_draw, phi_deg[idx_used], jump=180.0)
            head_phi_line.set_data(xw, yw); series_used.append(yw)
        if head_beta_line is not None:
            xw, yw = _xy_with_wrap_breaks(t_draw, beta_deg[idx_used], jump=180.0)
            head_beta_line.set_data(xw, yw); series_used.append(yw)
        if series_used:
            ycat = np.concatenate([y[np.isfinite(y)] for y in series_used]) if series_used else np.array([])
            if ycat.size:
                vmax = float(np.nanmax(np.abs(ycat)))
                target = max(5.0, 1.2 * vmax)
                head_ylim = (1.0 - HEAD_SMOOTH) * head_ylim + HEAD_SMOOTH * np.array([-target, target])
                ax_head.set_ylim(head_ylim[0], head_ylim[1])

        # steering (deg)
        if theta_hist is not None:
            th_used = theta_hist[idx_used]
            steer_line.set_data(t_draw, th_used)
            if th_max_deg is not None:
                ax_steer.set_ylim(-1.2 * th_max_deg, 1.2 * th_max_deg)
                xpair = [0.0, T_total]
                if steer_min_line is not None:
                    steer_min_line.set_data(xpair, [-th_max_deg, -th_max_deg])
                if steer_max_line is not None:
                    steer_max_line.set_data(xpair, [th_max_deg, th_max_deg])
            else:
                th_lim = max(5.0, 1.2 * np.nanmax(np.abs(th_used)))
                ax_steer.set_ylim(-th_lim, th_lim)
            s_avg = np.nanmean(th_used); s_min = np.nanmin(th_used); s_max = np.nanmax(th_used)
            steer_box.set_text(f"avg: {s_avg:4.1f}°\nmin: {s_min:4.1f}°\nmax: {s_max:4.1f}°")

        # solver time (ms)
        if solve_time is not None:
            st = np.asarray(solve_time)
            iu = idx_used[idx_used < st.size]
            td_ = t_axis[iu]
            ms = 1000.0 * st[iu]
            solv_line.set_data(td_, ms)
            ax_solv.set_ylim(0, max(10.0, 1.1 * np.nanmax(ms)))
            s_avg = np.nanmean(ms); s_min = np.nanmin(ms); s_max = np.nanmax(ms)
            solv_box.set_text(f"avg: {s_avg:4.1f} ms\nmin: {s_min:4.1f} ms\nmax: {s_max:4.1f} ms")

        # info text (under legend)
        info_txt.set_text(_format_laptimes(t_axis[i]))

        return (line_hist, car_rect, corr_line, line_horz, sc,
                vel_ref_line, vel_act_line, vel_pred_line, vel_min_line, vel_max_line,
                cte_line, cte_lo_line, cte_hi_line,
                D_line, D_min_line, D_max_line,
                (head_psi_line if head_psi_line is not None else ax_head),
                (head_phi_line if head_phi_line is not None else ax_head),
                (head_beta_line if head_beta_line is not None else ax_head),
                steer_line, solv_line, cost_line, info_txt, *obs_patches)

    anim = FuncAnimation(
        fig, update, frames=len(draw_idx),
        interval=1000.0/fps, blit=False, repeat=False
    )

    if save_path:
        print(f"Saving animation to: {save_path}")
        anim.save(save_path, fps=fps, dpi=140)
    else:
        plt.show()

    if (mpc is not None) and hasattr(mpc, "clear_corridor_shifts"):
        mpc.clear_corridor_shifts()
    return anim, fig
