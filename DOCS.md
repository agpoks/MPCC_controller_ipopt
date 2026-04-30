# MPCC Controller — Technical Documentation

## Overview

This project implements a **Model Predictive Contouring Controller (MPCC)** for a 1/10-scale autonomous RC racing car.  The controller solves a nonlinear programme (NLP) at every time step using **IPOPT** with the **HSL MA57** sparse direct linear solver for speed.

The car follows a pre-computed racing line, avoids moving circular obstacles, and handles obstacle-induced deceleration + overtaking automatically.

---

## System Architecture

```
params/vehicle.yaml          Physical parameters (mass, tyre, axle lengths …)
params/mpcc_tuning.yaml      Cost weights, horizon, IPOPT settings

main.cpp ─── YAML loading ─┐
              Scenario loop │   for each scenario:
              run_scenario()│     MPC::set_initial_params()
                            │     MPC::setup_MPC()          (build NLP once)
                            │     for k = 0 … STEPS:
                            │       MPC::solve(x)           (one IPOPT call)
                            │       MPC::propagate(x,u)     (RK4 simulation)
                            │     save_run_csv_auto()        (CSV logs)
                            │     playback_dashboard_csv.py  (GIF)
                            └──────────────────────────────────────────────

MPCC_controller.cpp          NLP formulation, warm start, fallback
rl_manager.cpp               B-spline LUTs for track geometry
utils.cpp                    MovingObstacle, arc-length re-estimation, CSV I/O
scripts/playback_dashboard_csv.py  Playback + GIF export
scripts/viz.py               Animation helpers
```

---

## Vehicle Model

### State vector  `x ∈ ℝ¹⁰`

| idx | symbol | meaning | unit |
|-----|--------|---------|------|
| 0 | X | global x-position | m |
| 1 | Y | global y-position | m |
| 2 | φ | heading angle | rad |
| 3 | v_x | longitudinal body-frame velocity | m/s |
| 4 | v_y | lateral body-frame velocity | m/s |
| 5 | r | yaw rate | rad/s |
| 6 | s | arc-length progress | m |
| 7 | D | drivetrain command (normalised) | – |
| 8 | δ | front steering angle | rad |
| 9 | v_s | virtual progress speed ds/dt | m/s |

### Input vector  `u ∈ ℝ³`

| idx | symbol | meaning |
|-----|--------|---------|
| 0 | dD | rate of change of D |
| 1 | dδ | rate of change of δ |
| 2 | dv_s | rate of change of v_s |

Making D, δ, v_s **actuator states** (integrated from rates) is a standard MPCC trick: it avoids discontinuities at the actuator limits and automatically adds smoothing because the rate is penalised in the cost.

### Tyre forces (simplified Pacejka)

```
α_f = δ − atan2(v_y + l_f·r, v_x)     front slip angle
α_r =    − atan2(v_y − l_r·r, v_x)     rear  slip angle

F_fy = D_f · sin( C_f · atan( B_f · α_f ) )   front lateral
F_ry = D_r · sin( C_r · atan( B_r · α_r ) )   rear  lateral
F_rx = C_m1·D − C_m2·D·v_x              longitudinal traction
F_fric = −C_r0 − C_r2·v_x²              rolling resistance
```

### Dynamics (body frame)

```
v̇_x = ( F_rx + F_fric − F_fy·sin(δ) + m·v_y·r ) / m
v̇_y = ( F_ry + F_fy·cos(δ) − m·v_x·r ) / m
ṙ   = ( F_fy·l_f·cos(δ) − F_ry·l_r ) / I_z
Ẋ   = v_x·cos(φ) − v_y·sin(φ)
Ẏ   = v_x·sin(φ) + v_y·cos(φ)
φ̇   = r
ṡ   = v_s        (virtual arc-length)
Ḋ   = dD
δ̇   = dδ
v̇_s = dv_s
```

### Integration

| Use | Method | Reason |
|-----|--------|--------|
| NLP constraints | Forward Euler | ~4× smaller expression graph vs. RK4 → faster Jacobian |
| Simulation loop | RK4 | Accuracy over long runs |

The model mismatch (Euler vs. RK4) is corrected every step by re-estimating `s` from the actual (X,Y) position.

---

## NLP Formulation

### Decision variables

```
X ∈ ℝ^{NX × (N+1)}   state trajectory
U ∈ ℝ^{NU × N}       input sequence
S_cor ∈ ℝ^N          soft corridor slack  (≥ 0)
S_obs ∈ ℝ^{n_obs × N} hard obstacle slack (≥ 0, optional)
```

### Cost function

```
J = Σ_{k=0}^{N-1} [
      q_C · e_c²              contouring error (lateral)
    + q_L · e_l²              lag error (longitudinal)
    − q_vs · v_s             progress reward
    + q_vref · (v_s − v_ref)² speed-reference tracking
    + r_D  · (dD_k − dD_{k-1})²   throttle smoothing
    + r_δ  · (dδ_k − dδ_{k-1})²   steering smoothing
    + r_vs · (dv_s_k − dv_s_{k-1})² progress-rate smoothing
    + q_slip · max(0, |v_y| − v_y^soft)²   slip constraint
    + q_fric · max(0, (F_rx/F_peak)² + (F_ry/F_peak)² − 1)²  friction ellipse
    + obstacle_penalty(X_{k+1}, Y_{k+1})
    + q_obs_hard · S_obs_{k}²
    + q_cor · S_cor_{k}²
  ]
  + q_CN · [q_C · e_{c,N}² + q_L · e_{l,N}²]    terminal cost
```

### Contouring and lag errors

The Frenet frame at arc-length `s` has:
- tangent direction `(t_x, t_y)` (unit, along the racing line)
- normal direction `(−t_y, t_x)` (pointing left)

```
e_c = −t_y·(X−c_x) + t_x·(Y−c_y)    contouring error (+  = left of racing line)
e_l =  t_x·(X−c_x) + t_y·(Y−c_y)    lag error        (+  = behind virtual ref)
```

### Corridor constraint (soft)

```
e_c_r − S_cor ≤ e_c ≤ e_c_l + S_cor
S_cor ≥ 0
cost += q_cor · S_cor²
```

`e_c_l` and `e_c_r` are the signed lateral offsets of the left and right track boundaries.  The slack `S_cor ≥ 0` keeps the NLP always feasible even when the car is already outside the corridor.

### Obstacle constraint (soft + optional hard-slack)

For obstacle `i` with position `(ox, oy)` and radius `r_obs`:

```
soft:  cost += q_obs · max(0, r_req² − dist²)² / r_req²
hard:  dist² + S_obs ≥ r_req²,   S_obs ≥ 0
       cost += q_obs_hard · S_obs²
where  r_req = r_obs + r_car + margin
```

The soft penalty activates when the car enters the forbidden circle; the hard-slack constraint adds an additional barrier while preserving feasibility.

### Friction ellipse (soft)

Without combined-slip correction, the simplified Pacejka model allows simultaneous peak longitudinal and lateral forces, which would cause real-car spin-out.  This soft penalty discourages such operating points:

```
combined = (F_rx / F_peak)² + (F_ry / F_peak)²
cost += q_fric · max(0, combined − 1)²
```

### Lateral slip (soft)

Large `|v_y|` indicates tyre saturation.  A soft upper bound keeps the predicted trajectory within a reasonable slip regime:

```
cost += q_slip · max(0, |v_y| − v_y^soft)²
```

### Parameters (numeric, updated before each solve)

| Parameter | Dimension | Contents |
|-----------|-----------|----------|
| `p_x0` | NX | current measured state |
| `p_obs` | n_obs × 3 | [x, y, radius] per obstacle |
| `p_u_prev` | NU | previous applied input (for rate cost) |
| `p_track` | (N+1) × 8 | [cx,cy,tx,ty,lx,ly,rx,ry] at each horizon step |
| `p_vref` | N+1 | curvature-based reference speed at each step |

The track geometry `p_track` is evaluated **numerically** before each IPOPT call.  This avoids embedding symbolic B-spline nodes inside the NLP graph, reducing the Jacobian computation time significantly.

---

## Solver: IPOPT + HSL MA57

IPOPT is a primal-dual interior-point method for general NLPs.  MA57 (from the HSL Mathematical Software Library) provides a fast sparse direct linear solver for the KKT system at each iteration.

| Setting | Default | Real-time (50 Hz) |
|---------|---------|-------------------|
| `N` | 15 | 10 |
| `dt` [s] | 0.05 | 0.02 |
| `ipopt_max_iter` | 120 | 60 |
| `ipopt_tol` | 1e-4 | 1e-3 |
| `ipopt_acceptable_tol` | 1e-3 | 5e-3 |
| `ipopt_max_cpu_time` [s] | 0 (off) | 0.018 |

Warm starting (`warm_start_init_point = yes`) reuses the shifted previous solution as the initial primal-dual guess, typically halving solve time after the first step.

### Warm start invalidation

If the arc-length `s` jumps by more than 35% of `s_total` between steps (lap wrap), the warm start is discarded and a cold start is used.  This prevents IPOPT from using a warm start with wrong arc-length values after the car completes a lap.

### Fallback on failure

```
solve ok        → use IPOPT solution, update warm start
solve failed + warm start exists  → apply shifted warm start (previous plan)
solve failed, no warm start       → apply cold-start equilibrium (zero rates)
```

---

## Track Representation

Track geometry is read from CSV files in `raceline/`:

| File | Contents |
|------|----------|
| `centerline_waypoints.csv` | (x, y) pairs along the racing line |
| `left_waypoints.csv` | left boundary |
| `right_waypoints.csv` | right boundary |
| `center_spline_derivatives.csv` | unit tangent vectors at each centre-line point |

All waypoints are extended by ~50% (wrap around to the start) and fitted with B-splines parameterised by arc-length.  The extension allows the NLP to look ahead past the start/finish line without discontinuities.

### Reference speed

A curvature-based reference speed is computed by `RLManager::build_reference_speed()`:

```
κ(s) = |t_x · ṫ_y − t_y · ṫ_x|     curvature at s
v_ref(s) = v_max − α(s) · (v_max − v_min)
α(s) = min(1, κ(s) / κ_ref)          κ_ref = 95th percentile
```

High curvature → low reference speed; straight sections → full speed.

---

## Obstacle Avoidance and Overtaking

### Moving obstacles

`MovingObstacle` maintains arc-length position `s` and speed `vs` along the track centre-line with a lateral offset.  At each simulation step the obstacle is stepped forward by `vs · dt` and its (x, y) world coordinates are computed from the centre-line B-splines.

### How overtaking emerges

The MPCC does not have an explicit overtaking manoeuvre.  It emerges from the combination of:
1. **Obstacle penalty / hard-slack** push the planned trajectory away from the obstacle.
2. **Corridor constraint** keeps the trajectory inside the track.
3. **Progress reward (−q_vs · vs)** drives the car to maximise forward progress.

When the obstacle is in a section narrow enough that neither side provides enough clearance, the MPCC reduces `vs` (slows down the virtual reference) and falls behind.  When the track widens, the obstacle penalty resolves to a path that goes around the obstacle, and the car overtakes.

---

## Configuration Files

### `params/vehicle.yaml`

Vehicle physical parameters.  Change these to match your specific car.  No rebuild required.

```yaml
Cm1: 0.287        # drivetrain gain [N per unit D]
Cm2: 0.0545       # speed-dependent drivetrain loss
Cr0: 0.0518       # constant rolling resistance
Cr2: 0.00035      # quadratic rolling resistance
Bf: 2.579         # front Pacejka stiffness factor
Cf: 1.2           # front Pacejka shape factor
Df: 0.192         # front peak lateral force [N]
...
```

### `params/mpcc_tuning.yaml`

All MPCC cost weights and IPOPT settings.  No rebuild required.

Key parameters for tuning:

| Parameter | Effect |
|-----------|--------|
| `mpc_w_cte` ↑ | Tighter lateral tracking |
| `mpc_w_lag` ↑ | Penalises falling behind racing line |
| `mpc_w_p` ↑ | More aggressive progress |
| `mpc_w_vref` ↑ | Follows curvature-based speed profile more closely |
| `mpc_w_slip` ↑ | Less lateral sliding (drifting) |
| `mpc_w_friction` ↑ | Less combined-force overload |
| `mpc_w_delta_d` ↑ | Smoother steering |

---

## Running

```bash
# Build and run both scenarios (GIFs generated automatically)
./run.sh

# Force clean rebuild
./run.sh rebuild

# Playback a specific result interactively
python3 scripts/playback_dashboard_csv.py \
    --results-dir results/scenario_standard_<timestamp> \
    --track-folder raceline

# Export GIF manually
python3 scripts/playback_dashboard_csv.py \
    --results-dir results/scenario_standard_<timestamp> \
    --gif plots/playback.gif --fps 15 --no-show
```

---

## Outputs

Each scenario produces a timestamped folder under `results/`:

```
results/scenario_<name>_<timestamp>/
├── states_ctrls.csv    per-step state, control, cost, solve time
├── predictions.csv     per-step MPC horizon (x, y, psi, s)
├── obstacles.csv       per-step obstacle positions
├── u_pred.csv          per-step predicted vs over horizon
├── playback.gif        animated dashboard
└── plots/
    ├── corridor.png    XY trajectory with track boundaries and obstacles
    ├── outputs.png     state time series (x, y, ψ, s, v_x)
    ├── inputs.png      actuator states (D, δ, vs)
    ├── timing.png      IPOPT solve time per step
    ├── progress.png    arc-length and speed over time
    ├── cost.png        MPC objective value over time
    └── cte.png         cross-track error over time
```

### Playback dashboard panels

| Panel | Contents |
|-------|----------|
| Track view | Car footprint, driven path, predicted horizon, current corridor, moving obstacles |
| Acceleration | Normalised a_lat / a_long scatter (friction circle check) |
| Velocity | Simulated speed, reference speed, predicted horizon speed |
| Lateral deviation | Cross-track error with corridor bounds |
| Drivetrain D | D command with hard limits |
| Heading | ψ (heading), φ (track tangent), β = ψ−φ (slip heading error) |
| Steering θ | Front steering angle with limits |
| Solver time | IPOPT wall-clock time per step [ms] |
| Lap timing | Current lap, best lap, last lap (auto-detected from s crossing s=0) |

---

## Known Limitations

1. **Simplified Pacejka without combined slip**: The tyre model does not reduce lateral force when longitudinal force is high.  The friction-ellipse soft penalty mitigates but does not fully resolve this.  A combined-slip extension (e.g. Pacejka MF6.1) would eliminate the issue at the cost of a more complex NLP.

2. **Euler in NLP vs. RK4 in simulation**: The NLP uses first-order Euler integration for speed.  The mismatch is corrected by re-estimating `s` from (X,Y) at every step, but there is a small systematic error in the predicted trajectory.

3. **Static warm start invalidation on lap wrap**: When `s` resets, a cold start is used.  In practice this adds ~100 ms on the first step of each new lap.

4. **No explicit combined-lane planning**: Overtaking is an emergent behaviour from the obstacle and corridor costs.  Complex scenarios (e.g. two simultaneous large obstacles in a narrow section) may cause the MPCC to stall.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| CasADi (≥ 3.6) | Symbolic NLP formulation, automatic differentiation |
| IPOPT | Nonlinear interior-point solver |
| HSL MA57 | Sparse direct linear solver (faster than default MUMPS) |
| Python 3 + NumPy + Matplotlib + Pillow | Playback visualisation and GIF export |

HSL must be installed separately (academic/commercial licence).  Place `libcoinhsl.so` in `~/ThirdParty-HSL/.libs/`.  The controller auto-detects this path.
