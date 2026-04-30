#include "MPCC_controller.h"

#include <chrono>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <cstring>

using namespace casadi;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

// Evaluate a 1-D CasADi interpolant numerically.
static double lut_num(const Function& f, double s)
{
    return static_cast<double>(f(std::vector<DM>{DM(s)}).at(0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Dynamics: single-track planar model with Pacejka simplified tyres
//
// Builds two CasADi functions:
//   f_cont_  : x_dot = f(x, u)                — used in NLP (Euler step)
//   f_disc_  : x_{k+1} = f_rk4(x_k, u_k, dt) — used in propagate() (RK4)
//
// Using Euler in the NLP halves the symbolic expression size vs. RK4 and
// gives a 2-3× speed-up for IPOPT Jacobian/Hessian evaluation at acceptable
// accuracy for dt = 0.05 s.  The simulation always uses RK4 for fidelity.
// ─────────────────────────────────────────────────────────────────────────────
void MPC::build_dynamics()
{
    using casadi::SX;

    SX x  = SX::sym("x",  NX);
    SX u  = SX::sym("u",  NU);
    SX dt = SX::sym("dt", 1);

    // State extraction
    SX phi   = x(2);
    SX vx    = x(3), vy = x(4), r = x(5);
    SX D     = x(7), delta = x(8), vs = x(9);

    // Input extraction
    SX dD = u(0), dDelta = u(1), dVs = u(2);

    // Regularisation to prevent division by zero at vx ≈ 0
    const double eps = 1e-3;

    // Pacejka slip angles (linearised atan2)
    SX alpha_f = delta - atan((vp_.lf * r + vy) / (vx + eps));
    SX alpha_r =        -atan((vy - vp_.lr * r) / (vx + eps));

    // Lateral tyre forces (simplified Pacejka — no combined-slip correction)
    SX Ffy = vp_.Df * sin(vp_.Cf * atan(vp_.Bf * alpha_f));
    SX Fry = vp_.Dr * sin(vp_.Cr * atan(vp_.Br * alpha_r));

    // Longitudinal traction and rolling resistance
    SX Frx   = vp_.Cm1 * D - vp_.Cm2 * D * vx;
    SX Ffric = -vp_.Cr0 - vp_.Cr2 * pow(vx, 2);

    // Body-frame velocity dynamics (Newton / Euler equations)
    SX vx_dot = (Frx + Ffric - Ffy * sin(delta) + vp_.m * vy * r) / vp_.m;
    SX vy_dot = (Fry + Ffy * cos(delta) - vp_.m * vx * r) / vp_.m;
    SX r_dot  = (Ffy * vp_.lf * cos(delta) - Fry * vp_.lr) / vp_.Iz;

    // Global position kinematics
    SX X_dot   = vx * cos(phi) - vy * sin(phi);
    SX Y_dot   = vx * sin(phi) + vy * cos(phi);
    SX phi_dot = r;

    // Arc-length and actuator-state dynamics (integrator states)
    SX s_dot     = vs;
    SX D_dot     = dD;
    SX delta_dot = dDelta;
    SX vs_dot    = dVs;

    SX f_expr = vertcat(std::vector<SX>{
        X_dot, Y_dot, phi_dot,
        vx_dot, vy_dot, r_dot,
        s_dot, D_dot, delta_dot, vs_dot});

    f_cont_ = Function("f_cont",
                       std::vector<SX>{x, u},
                       std::vector<SX>{f_expr},
                       {"x","u"}, {"xdot"});

    Function f_cont = f_cont_;  // local alias for RK4 closures below

    // RK4 integration for propagate() (simulation loop uses this)
    auto call = [&](const SX& xx, const SX& uu) -> SX {
        return f_cont(std::vector<SX>{xx, uu})[0];
    };
    SX k1 = call(x,                 u);
    SX k2 = call(x + dt*0.5*k1,    u);
    SX k3 = call(x + dt*0.5*k2,    u);
    SX k4 = call(x + dt*k3,        u);
    SX xn = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);

    f_disc_ = Function("f_disc",
                       std::vector<SX>{x, u, dt},
                       std::vector<SX>{xn},
                       {"x","u","dt"}, {"xip1"});
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter loading from param_num map
//
// Vehicle parameters use a "vp_" prefix so they can be loaded from
// params/vehicle.yaml alongside MPCC tuning values.
// ─────────────────────────────────────────────────────────────────────────────
void MPC::set_initial_params(const std::map<std::string, double>& num,
                              const std::map<std::string, std::string>& /*str*/)
{
    auto get = [&](const std::string& k, double def) {
        auto it = num.find(k);
        return (it != num.end()) ? it->second : def;
    };

    N_   = static_cast<int>(get("N",  15));
    dt_  = get("dT", 0.05);

    delta_max_ = get("theta_max", 0.35);
    delta_min_ = -delta_max_;
    theta_max  = delta_max_;

    vx_max_     = get("v_max",    3.5);
    vs_max_     = get("p_max",    vx_max_);
    vs_min_     = get("p_min",    0.0);
    s_ext_max_  = get("s_max",  300.0);
    car_radius_ = get("car_radius", 0.15);

    D_min_ = get("D_min", D_min_);
    D_max_ = get("D_max", D_max_);

    dD_max_     = get("dD_max",     15.0);
    dDelta_max_ = get("dDelta_max", 15.0);
    dVs_max_    = get("dVs_max",    10.0);

    max_obs_    = static_cast<int>(get("max_obstacles", 4));
    obs_margin_ = get("obs_margin", 0.27);

    // MPCC cost weights
    qC_       = get("mpc_w_cte",     1.0);
    qL_       = get("mpc_w_lag",   500.0);
    qVs_      = get("mpc_w_p",      0.05);
    qVref_    = get("mpc_w_vref",    3.0);
    rdD_      = get("mpc_w_accel",   1e-4);
    rdDelta_  = get("mpc_w_delta_d", 0.01);
    rdVs_     = get("mpc_w_delta_p", 1e-5);
    qObs_     = get("mpc_w_obs",   3000.0);
    qObsHardSlack_ = get("mpc_w_obs_hard_slack", 20000.0);
    use_hard_obs_slack_ = (get("mpc_use_hard_obs_slack", 1.0) > 0.5);

    // Friction / slip soft constraints
    qSlip_              = get("mpc_w_slip",        50.0);
    vy_soft_            = get("vy_soft_limit",      0.25);
    qFriction_          = get("mpc_w_friction",    30.0);
    friction_long_peak_ = get("friction_long_peak", 0.0);
    friction_lat_peak_  = get("friction_lat_peak",  0.0);

    // IPOPT settings
    ipopt_max_iter_       = static_cast<int>(get("ipopt_max_iter", 120));
    ipopt_tol_            = get("ipopt_tol",            1e-4);
    ipopt_acceptable_tol_ = get("ipopt_acceptable_tol", 1e-3);
    ipopt_acceptable_obj_change_tol_ = get("ipopt_acceptable_obj_change_tol", 1e-3);
    ipopt_max_cpu_time_   = get("ipopt_max_cpu_time", 0.0);

    // ── Vehicle parameters (vp_ prefix in map) ───────────────────────────
    vp_.Cm1 = get("vp_Cm1", vp_.Cm1);
    vp_.Cm2 = get("vp_Cm2", vp_.Cm2);
    vp_.Cr0 = get("vp_Cr0", vp_.Cr0);
    vp_.Cr2 = get("vp_Cr2", vp_.Cr2);
    vp_.Bf  = get("vp_Bf",  vp_.Bf);
    vp_.Cf  = get("vp_Cf",  vp_.Cf);
    vp_.Df  = get("vp_Df",  vp_.Df);
    vp_.Br  = get("vp_Br",  vp_.Br);
    vp_.Cr  = get("vp_Cr_tire", vp_.Cr);  // Cr_tire in YAML to avoid YAML key clash
    vp_.Dr  = get("vp_Dr",  vp_.Dr);
    vp_.m   = get("vp_m",   vp_.m);
    vp_.Iz  = get("vp_Iz",  vp_.Iz);
    vp_.lf  = get("vp_lf",  vp_.lf);
    vp_.lr  = get("vp_lr",  vp_.lr);

    // ── HSL linear solver configuration ──────────────────────────────────
    // Priority: env-var override → auto-detect from common install paths.
    const char* env_solver = std::getenv("MPC_IPOPT_LINEAR_SOLVER");
    if (env_solver && std::strlen(env_solver) > 0)
        ipopt_linear_solver_ = std::string(env_solver);

    const char* env_hsllib = std::getenv("MPC_IPOPT_HSL_LIB");
    if (env_hsllib && std::strlen(env_hsllib) > 0) {
        ipopt_hsllib_ = std::string(env_hsllib);
    } else {
        for (auto var : {"IPOPT_HSL_LIB", "HSL_LIB"}) {
            const char* v = std::getenv(var);
            if (v && std::strlen(v) > 0) { ipopt_hsllib_ = v; break; }
        }
    }

    if (ipopt_hsllib_.empty()) {
        namespace fs = std::filesystem;
        const char* home = std::getenv("HOME");
        if (home && std::strlen(home) > 0) {
            for (auto& cand : { std::string(home) + "/ThirdParty-HSL/.libs/libcoinhsl.so",
                                 std::string(home) + "/ThirdParty-HSL/.libs/libhsl.so" }) {
                if (fs::exists(cand)) { ipopt_hsllib_ = cand; break; }
            }
        }
    }

    // Extend LD_LIBRARY_PATH so dlopen can find the HSL shared library.
    if (!ipopt_hsllib_.empty()) {
        namespace fs = std::filesystem;
        if (fs::exists(ipopt_hsllib_)) {
            const std::string dir = fs::path(ipopt_hsllib_).parent_path().string();
            const char* ld = std::getenv("LD_LIBRARY_PATH");
            std::string cur = ld ? std::string(ld) : "";
            if (cur.find(dir) == std::string::npos) {
                setenv("LD_LIBRARY_PATH", (dir + (cur.empty() ? "" : ":" + cur)).c_str(), 1);
            }
        }
    }

    u_prev_ = DM::zeros(NU);
}

// ─────────────────────────────────────────────────────────────────────────────
// Track data: B-spline LUTs for centre-line, boundaries, and reference speed
// ─────────────────────────────────────────────────────────────────────────────
void MPC::set_track_data(const Function& c_lut_x, const Function& c_lut_y,
                          const Function& c_lut_dx, const Function& c_lut_dy,
                          const Function& r_lut_x, const Function& r_lut_y,
                          const Function& l_lut_x, const Function& l_lut_y,
                          const std::vector<double>& s_ext, double s_total,
                          const Function& v_ref_lut)
{
    c_lut_x_  = c_lut_x;  c_lut_y_  = c_lut_y;
    c_lut_dx_ = c_lut_dx; c_lut_dy_ = c_lut_dy;
    r_lut_x_  = r_lut_x;  r_lut_y_  = r_lut_y;
    l_lut_x_  = l_lut_x;  l_lut_y_  = l_lut_y;
    v_ref_lut_ = v_ref_lut;
    s_total_   = s_total;
    if (!s_ext.empty()) s_ext_max_ = s_ext.back();
}

// ─────────────────────────────────────────────────────────────────────────────
// Boundary orientation check
//
// The corridor constraint requires e_c_left > 0 in the signed-distance frame.
// Sample a few points around the track and swap left/right if the sign is wrong.
// ─────────────────────────────────────────────────────────────────────────────
void MPC::check_and_fix_boundary_orientation()
{
    constexpr int N_CHECK = 8;
    int vote = 0;
    for (int i = 0; i < N_CHECK; i++) {
        double s  = s_total_ * (i + 0.5) / N_CHECK;
        double cx = lut_num(c_lut_x_, s), cy = lut_num(c_lut_y_, s);
        double tx = lut_num(c_lut_dx_, s), ty = lut_num(c_lut_dy_, s);
        double lx = lut_num(l_lut_x_, s), ly = lut_num(l_lut_y_, s);
        // Left normal in track frame: n = [-ty, tx].  e_c_l = n^T*(l−c)
        if (-ty*(lx-cx) + tx*(ly-cy) < 0) vote++;
    }
    if (vote > N_CHECK / 2) {
        std::swap(l_lut_x_, r_lut_x_);
        std::swap(l_lut_y_, r_lut_y_);
        std::cout << "[MPCC] Auto-corrected left/right boundary orientation.\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Obstacle soft penalty
//
// For each obstacle i the penalty activates when the predicted position enters
// the combined radius (obstacle + car + safety margin).  Using a squared
// ReLU keeps the expression smooth for IPOPT's gradient-based solve.
// ─────────────────────────────────────────────────────────────────────────────
MX MPC::obstacle_penalty(const MX& Xk, const MX& Yk) const
{
    const double min_sep = car_radius_ + obs_margin_;
    MX pen = MX::zeros(1);
    for (int i = 0; i < max_obs_; i++) {
        MX ox  = p_obs_(i, 0), oy  = p_obs_(i, 1), or_ = p_obs_(i, 2);
        MX r_req_sq  = pow(or_ + min_sep, 2);
        MX dist_sq   = pow(Xk - ox, 2) + pow(Yk - oy, 2);
        // Positive excess when inside the forbidden circle
        MX excess    = r_req_sq - dist_sq;
        MX pos_exc   = (excess + fabs(excess)) / 2.0;
        pen += qObs_ * pow(pos_exc, 2) / (r_req_sq + 1e-6);
    }
    return pen;
}

// ─────────────────────────────────────────────────────────────────────────────
// IPOPT solver configuration
// ─────────────────────────────────────────────────────────────────────────────
void MPC::configure_solver()
{
    Dict ipopt_opts;
    ipopt_opts["max_iter"]                  = ipopt_max_iter_;
    ipopt_opts["tol"]                       = ipopt_tol_;
    ipopt_opts["acceptable_tol"]            = ipopt_acceptable_tol_;
    ipopt_opts["acceptable_obj_change_tol"] = ipopt_acceptable_obj_change_tol_;
    ipopt_opts["print_level"]               = 0;
    ipopt_opts["warm_start_init_point"]     = std::string("yes");
    ipopt_opts["warm_start_bound_push"]     = 1e-6;
    ipopt_opts["warm_start_mult_bound_push"]= 1e-6;
    ipopt_opts["warm_start_slack_bound_push"]= 1e-6;
    ipopt_opts["mu_init"]                   = 1e-3;
    ipopt_opts["mu_strategy"]               = std::string("adaptive");
    ipopt_opts["nlp_scaling_method"]        = std::string("gradient-based");
    ipopt_opts["fixed_variable_treatment"]  = std::string("relax_bounds");
    ipopt_opts["linear_solver"]             = ipopt_linear_solver_;
    if (ipopt_max_cpu_time_ > 0.0)
        ipopt_opts["max_cpu_time"] = ipopt_max_cpu_time_;

    Dict solver_opts;
    solver_opts["ipopt"]      = ipopt_opts;
    solver_opts["verbose"]    = false;
    solver_opts["print_time"] = false;
    solver_opts["expand"]     = true;  // inline SX → faster AD

    opti_.solver("ipopt", solver_opts);
}

// ─────────────────────────────────────────────────────────────────────────────
// Build the NLP — called once before the simulation loop.
//
// Decision variables:
//   X_    : NX × (N+1) state trajectory
//   U_    : NU × N     input sequence
//   S_cor :  1 × N     soft corridor slack (≥ 0)
//   S_obs : max_obs × N  hard obstacle slack (≥ 0, optional)
//
// Parameters (updated numerically before each solve):
//   p_x0_    : current state (NX)
//   p_obs_   : obstacle matrix (max_obs × 3)
//   p_u_prev_: previous input (NU)
//   p_track_ : track geometry at horizon steps ((N+1) × 8)
//   p_vref_  : reference speed at each step (N+1)
// ─────────────────────────────────────────────────────────────────────────────
void MPC::setup_MPC()
{
    check_and_fix_boundary_orientation();
    build_dynamics();

    opti_ = Opti();

    // Decision variables
    X_       = opti_.variable(NX, N_ + 1);
    U_       = opti_.variable(NU, N_);
    MX S_cor = opti_.variable(1,  N_);   // soft corridor slack per horizon step

    MX S_obs;
    if (use_hard_obs_slack_)
        S_obs = opti_.variable(max_obs_, N_);

    // Parameters
    p_x0_     = opti_.parameter(NX);
    p_obs_    = opti_.parameter(max_obs_, 3);
    p_u_prev_ = opti_.parameter(NU);
    p_vref_   = opti_.parameter(N_ + 1);
    p_track_  = opti_.parameter(N_ + 1, 8);  // [cx,cy,tx,ty,lx,ly,rx,ry] per step

    // Slack non-negativity
    opti_.subject_to(S_cor >= 0.0);
    if (use_hard_obs_slack_)
        opti_.subject_to(opti_.bounded(0.0, S_obs,
                                        std::numeric_limits<double>::infinity()));

    // Fix initial state to the measured x0
    opti_.subject_to(X_(Slice(), 0) == p_x0_);

    MX total_cost = MX::zeros(1);

    for (int k = 0; k < N_; k++) {
        MX xk   = X_(Slice(), k);
        MX uk   = U_(Slice(), k);
        MX xkp1 = X_(Slice(), k+1);

        MX Xk  = xk(0), Yk = xk(1), vsk = xk(9);
        MX vxk = xk(3), vyk = xk(4), rk  = xk(5);
        MX Dk  = xk(7), deltak = xk(8);

        // Track geometry from numeric parameter (avoids symbolic LUT nodes,
        // which would add ~8*(N+1) B-spline sub-graphs to the NLP).
        MX cx = p_track_(k,0), cy = p_track_(k,1);
        MX tx = p_track_(k,2), ty = p_track_(k,3);
        MX lx = p_track_(k,4), ly = p_track_(k,5);
        MX rx = p_track_(k,6), ry = p_track_(k,7);

        // Signed contouring / lag errors in the Frenet frame
        MX epX = Xk - cx, epY = Yk - cy;
        MX e_l = tx*epX + ty*epY;           // lag   error (positive = behind ref)
        MX e_c = -ty*epX + tx*epY;          // contouring error (positive = left)

        // ── Stage cost ──────────────────────────────────────────────────
        MX uk_prev = (k == 0) ? MX(p_u_prev_) : MX(U_(Slice(), k-1));
        MX du = uk - uk_prev;

        total_cost += qC_    * pow(e_c, 2)          // contouring (lateral)
                   +  qL_    * pow(e_l, 2)          // lag (longitudinal)
                   -  qVs_   * vsk                  // reward forward progress
                   +  qVref_ * pow(vsk - p_vref_(k), 2)  // speed reference
                   +  rdD_   * pow(du(0), 2)        // throttle-rate smoothing
                   +  rdDelta_* pow(du(1), 2)       // steering-rate smoothing
                   +  rdVs_  * pow(du(2), 2);       // progress-rate smoothing

        // ── Friction ellipse soft constraint ────────────────────────────
        // Penalises combined rear-tyre loading (longitudinal + lateral)
        // exceeding the approximate friction limit.  Without this the
        // simplified Pacejka model allows simultaneous peak forces in both
        // directions, causing the car to spin out.
        if (qFriction_ > 0.0) {
            const double eps = 1e-3;
            MX alpha_r  = -atan((vyk - vp_.lr * rk) / (vxk + eps));
            MX Fry_k    = vp_.Dr * sin(vp_.Cr * atan(vp_.Br * alpha_r));
            MX Frx_k    = vp_.Cm1 * Dk - vp_.Cm2 * Dk * vxk;
            // Normalise by configured peaks; auto-detect if not set (0 = auto).
            // Using separate long/lat peaks avoids false violations when Cm1 >> Dr.
            const double fp_long = (friction_long_peak_ > 0.0) ? friction_long_peak_ : (vp_.Cm1 + 1e-6);
            const double fp_lat  = (friction_lat_peak_  > 0.0) ? friction_lat_peak_  : (vp_.Dr  + 1e-6);
            MX combined = pow(Frx_k / fp_long, 2) + pow(Fry_k / fp_lat, 2);
            MX excess   = (combined - MX(1.0) + fabs(combined - MX(1.0))) / 2.0;
            total_cost += qFriction_ * pow(excess, 2);
        }

        // ── Lateral slip (vy) soft constraint ───────────────────────────
        // Large |vy| indicates tyre saturation; penalise values above the
        // soft limit to discourage high-slip trajectories.
        if (qSlip_ > 0.0) {
            MX vy_exc = (fabs(vyk) - MX(vy_soft_) + fabs(fabs(vyk) - MX(vy_soft_))) / 2.0;
            total_cost += qSlip_ * pow(vy_exc, 2);
        }

        // ── Euler dynamics constraint ────────────────────────────────────
        // Forward-Euler is 4× cheaper than RK4 in expression graph size.
        // Accuracy is sufficient at dt = 0.05 s.
        MX xdot = f_cont_(std::vector<MX>{xk, uk}).at(0);
        opti_.subject_to(xkp1 == xk + dt_ * xdot);

        // ── Control bounds ───────────────────────────────────────────────
        opti_.subject_to(uk(0) >= -dD_max_);     opti_.subject_to(uk(0) <= dD_max_);
        opti_.subject_to(uk(1) >= -dDelta_max_); opti_.subject_to(uk(1) <= dDelta_max_);
        opti_.subject_to(uk(2) >= -dVs_max_);    opti_.subject_to(uk(2) <= dVs_max_);

        // ── Actuator state bounds on NEXT step ───────────────────────────
        // Only the integrator states (D, delta, vs, s) are hard-bounded.
        // Velocity states (vx, vy, r) are NOT hard-bounded in the NLP because
        // Euler dynamics + hard velocity bounds cause infeasibility near the
        // limits.  Their envelope is enforced by clamping in the simulation loop.
        MX D1  = xkp1(7), d1 = xkp1(8), vs1 = xkp1(9), s1 = xkp1(6);
        MX vx1 = xkp1(3);

        opti_.subject_to(D1  >= D_min_);     opti_.subject_to(D1  <= D_max_);
        opti_.subject_to(d1  >= delta_min_); opti_.subject_to(d1  <= delta_max_);
        opti_.subject_to(vs1 >= vs_min_);    opti_.subject_to(vs1 <= vs_max_);
        opti_.subject_to(s1  >= 0.0);        opti_.subject_to(s1  <= s_ext_max_);

        // Virtual speed coupling: vs ≤ vx + 0.5 prevents the optimizer from
        // advancing the virtual reference faster than the car can actually travel.
        opti_.subject_to(vs1 <= fmax(MX(0.0), vx1) + 0.5);

        // ── Obstacle penalties and hard-slack constraints on NEXT step ──
        MX X1 = xkp1(0), Y1 = xkp1(1);
        total_cost += obstacle_penalty(X1, Y1);

        if (use_hard_obs_slack_) {
            const double min_sep = car_radius_ + obs_margin_;
            for (int i = 0; i < max_obs_; ++i) {
                MX ox = p_obs_(i,0), oy = p_obs_(i,1), or_ = p_obs_(i,2);
                MX req_sq  = pow(or_ + min_sep, 2);
                MX dist_sq = pow(X1-ox, 2) + pow(Y1-oy, 2);
                MX sk_obs  = S_obs(i, k);
                // dist^2 + slack ≥ req^2  (hard barrier, slack ≥ 0 keeps feasibility)
                opti_.subject_to(opti_.bounded(
                    0.0, dist_sq + sk_obs - req_sq,
                    std::numeric_limits<double>::infinity()));
                total_cost += qObsHardSlack_ * pow(sk_obs, 2);
            }
        }

        // ── Soft corridor constraint on NEXT step ───────────────────────
        // Slack s_cor ≥ 0 keeps the NLP feasible when the car is already
        // outside the corridor (e.g. after a failed solve or large disturbance).
        MX cx1=p_track_(k+1,0), cy1=p_track_(k+1,1);
        MX tx1=p_track_(k+1,2), ty1=p_track_(k+1,3);
        MX lx1=p_track_(k+1,4), ly1=p_track_(k+1,5);
        MX rx1=p_track_(k+1,6), ry1=p_track_(k+1,7);
        MX e_c1   = -ty1*(X1-cx1) + tx1*(Y1-cy1);
        MX e_c_l1 = -ty1*(lx1-cx1) + tx1*(ly1-cy1);
        MX e_c_r1 = -ty1*(rx1-cx1) + tx1*(ry1-cy1);
        MX sk = S_cor(k);
        opti_.subject_to(e_c1 <= e_c_l1 + sk);
        opti_.subject_to(e_c1 >= e_c_r1 - sk);
        const double q_s = 500.0;
        total_cost += q_s * pow(sk, 2);

    }

    // ── Terminal cost ──────────────────────────────────────────────────────
    {
        MX xN = X_(Slice(), N_);
        MX cx = p_track_(N_,0), cy = p_track_(N_,1);
        MX tx = p_track_(N_,2), ty = p_track_(N_,3);
        MX epX = xN(0) - cx, epY = xN(1) - cy;
        MX e_l_N = tx*epX + ty*epY;
        MX e_c_N = -ty*epX + tx*epY;
        total_cost += qCN_mult_ * qC_ * pow(e_c_N, 2)
                    + qCN_mult_ * qL_ * pow(e_l_N, 2);
    }

    opti_.minimize(total_cost);
    cost_expr_ = total_cost;

    configure_solver();
    std::cout << "[MPCC] NLP built: N=" << N_ << " dt=" << dt_
              << " s | nx=" << NX << " nu=" << NU
              << " max_obs=" << max_obs_
              << " solver=" << ipopt_linear_solver_
              << " hsl=" << (ipopt_hsllib_.empty() ? "<auto>" : ipopt_hsllib_)
              << "\n";
}

void MPC::set_obstacles(const std::vector<Obstacle>& obs)
{
    obstacles_ = obs;
}

// ─────────────────────────────────────────────────────────────────────────────
// solve() — main entry point for each MPC step
//
// 1. Populates IPOPT parameters (obstacle matrix, track geometry).
// 2. Provides warm start (shifted previous solution) or cold start.
// 3. Calls IPOPT; on failure applies a safe fallback.
// 4. Extracts and clamps the first optimal input.
// ─────────────────────────────────────────────────────────────────────────────
MPC::Solution MPC::solve(const std::vector<double>& x0)
{
    // ── Obstacle parameter matrix (pad missing entries far away) ─────────
    DM obs_mat = DM::zeros(max_obs_, 3);
    for (int i = 0; i < max_obs_; i++) {
        if (i < static_cast<int>(obstacles_.size())) {
            obs_mat(i,0) = obstacles_[i].x;
            obs_mat(i,1) = obstacles_[i].y;
            obs_mat(i,2) = obstacles_[i].radius;
        } else {
            obs_mat(i,0) = 1e4;  // inactive obstacle, placed far away
            obs_mat(i,1) = 1e4;
            obs_mat(i,2) = 0.0;
        }
    }
    opti_.set_value(p_x0_,     DM(x0));
    opti_.set_value(p_obs_,    obs_mat);
    opti_.set_value(p_u_prev_, u_prev_);

    // ── Check for lap wrap and reset warm start if needed ─────────────────
    // When s resets from ~s_total to ~0 (new lap), the shifted warm start
    // contains large-s states that are incompatible with the new initial state.
    if (has_warm_start_ && X_warm_.size2() > 0) {
        double warm_s = static_cast<double>(X_warm_(6, 0));
        double new_s  = x0[6];
        if (std::abs(new_s - warm_s) > s_total_ * 0.35) {
            has_warm_start_ = false;  // force cold start after lap wrap
        }
    }

    // ── Track geometry parameters (evaluated numerically before each solve)
    // Using a numeric forward scan at vs_est avoids embedding B-spline symbolic
    // nodes in the NLP, reducing the expression graph size significantly.
    {
        DM track_vals = DM::zeros(N_ + 1, 8);
        DM vref_vals  = DM::zeros(N_ + 1);
        double s0 = x0[6];
        // Estimate look-ahead speed; clamp to at least 0.3 m/s so the geometry
        // points don't all cluster at the same arc-length when vs ≈ 0.
        double vs_est = std::max(0.3, std::clamp(x0[9], vs_min_,
                                                  std::min(x0[3] + 0.5, vs_max_)));

        for (int k = 0; k <= N_; k++) {
            double sk = std::min(s0 + k * dt_ * vs_est, s_ext_max_);
            track_vals(k,0) = lut_num(c_lut_x_,  sk);
            track_vals(k,1) = lut_num(c_lut_y_,  sk);
            track_vals(k,2) = lut_num(c_lut_dx_, sk);
            track_vals(k,3) = lut_num(c_lut_dy_, sk);
            track_vals(k,4) = lut_num(l_lut_x_,  sk);
            track_vals(k,5) = lut_num(l_lut_y_,  sk);
            track_vals(k,6) = lut_num(r_lut_x_,  sk);
            track_vals(k,7) = lut_num(r_lut_y_,  sk);
            vref_vals(k)    = v_ref_lut_.is_null() ? vx_max_ : lut_num(v_ref_lut_, sk);
        }
        opti_.set_value(p_track_, track_vals);
        opti_.set_value(p_vref_,  vref_vals);
    }

    // ── Cold-start generator ──────────────────────────────────────────────
    // Builds an equilibrium trajectory along the centre-line at current speed.
    auto make_cold_start = [&]() -> std::pair<DM, DM> {
        double s0   = x0[6];
        double vx0  = std::max(vx_min_, x0[3]);
        double denom = vp_.Cm1 - vp_.Cm2 * vx0;
        double D_eq  = std::clamp(
            (denom > 1e-6) ? (vp_.Cr0 + vp_.Cr2*vx0*vx0) / denom : 0.0,
            D_min_, D_max_);
        double vs0 = std::clamp(x0[9], vs_min_, std::min(vx0 + 0.5, vs_max_));

        DM Xi = DM::zeros(NX, N_ + 1);
        DM Ui = DM::zeros(NU, N_);
        for (int j = 0; j < NX; j++) Xi(j, 0) = x0[j];

        for (int k = 1; k <= N_; k++) {
            double sk = std::min(s0 + k * dt_ * vs0, s_ext_max_);
            Xi(0,k) = lut_num(c_lut_x_,  sk);
            Xi(1,k) = lut_num(c_lut_y_,  sk);
            Xi(2,k) = std::atan2(lut_num(c_lut_dy_, sk), lut_num(c_lut_dx_, sk));
            Xi(3,k) = vx0;
            Xi(4,k) = 0.0;
            Xi(5,k) = 0.0;
            Xi(6,k) = sk;
            Xi(7,k) = D_eq;
            Xi(8,k) = 0.0;
            Xi(9,k) = vs0;
        }
        return {Xi, Ui};
    };

    // ── Initial guess ─────────────────────────────────────────────────────
    if (has_warm_start_) {
        opti_.set_initial(X_, X_warm_);
        opti_.set_initial(U_, U_warm_);
    } else {
        auto [Xi, Ui] = make_cold_start();
        opti_.set_initial(X_, Xi);
        opti_.set_initial(U_, Ui);
    }

    // ── Call IPOPT ────────────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = true;
    try {
        opti_.solve();
    } catch (const std::exception& e) {
        ok = false;
        const std::string msg = e.what();
        std::cerr << "[MPCC] " << msg << "\n";

        // If IPOPT rejected an option (e.g. unsupported HSL solver name),
        // fall back to MUMPS once and retry.
        if (msg.find("Invalid_Option") != std::string::npos &&
            ipopt_linear_solver_ != "mumps") {
            std::cerr << "[MPCC] Falling back linear solver '"
                      << ipopt_linear_solver_ << "' → 'mumps'\n";
            ipopt_linear_solver_ = "mumps";
            configure_solver();
            try { opti_.solve(); ok = true; }
            catch (const std::exception& e2) {
                std::cerr << "[MPCC] Retry with mumps failed: " << e2.what() << "\n";
            }
        }
    }
    last_solve_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // ── Extract solution / fallback ────────────────────────────────────────
    // Priority:
    //   1. ok → use optimal values, update warm start
    //   2. !ok + warm start → apply shifted warm start (safe previous plan)
    //   3. !ok, no history → cold start (conservative, zero rates)
    Solution sol;
    sol.u_cmd_first.assign(NU, 0.0);

    if (ok) {
        DM Xsol = opti_.value(X_);
        DM Usol = opti_.value(U_);
        sol.X_opt = Xsol.T();
        sol.U_opt = Usol.T();
        try { last_cost = static_cast<double>(opti_.value(cost_expr_)); }
        catch (...) { last_cost = 0.0; }
        for (int j = 0; j < NU; j++)
            sol.u_cmd_first[j] = static_cast<double>(Usol(j, 0));
        // Shift the solution by one step for the next warm start
        X_warm_ = horzcat(Xsol(Slice(), Slice(1, N_+1)),
                          Xsol(Slice(), Slice(N_, N_+1)));
        U_warm_ = horzcat(Usol(Slice(), Slice(1, N_)),
                          Usol(Slice(), Slice(N_-1, N_)));
        has_warm_start_ = true;
    } else if (has_warm_start_) {
        // Use the previously computed (shifted) plan as a safe fallback.
        sol.X_opt = X_warm_.T();
        sol.U_opt = U_warm_.T();
        last_cost = 0.0;
        if (U_warm_.size2() > 0) {
            sol.u_cmd_first = { static_cast<double>(U_warm_(0,0)),
                                static_cast<double>(U_warm_(1,0)),
                                static_cast<double>(U_warm_(2,0)) };
        }
    } else {
        auto [Xi, Ui] = make_cold_start();
        sol.X_opt = Xi.T();
        sol.U_opt = Ui.T();
        last_cost = 0.0;
    }

    // Clamp first control to physical bounds before applying to the plant.
    sol.u_cmd_first[0] = std::clamp(sol.u_cmd_first[0], -dD_max_,     dD_max_);
    sol.u_cmd_first[1] = std::clamp(sol.u_cmd_first[1], -dDelta_max_, dDelta_max_);
    sol.u_cmd_first[2] = std::clamp(sol.u_cmd_first[2], -dVs_max_,    dVs_max_);

    u_prev_ = DM({sol.u_cmd_first[0], sol.u_cmd_first[1], sol.u_cmd_first[2]});
    return sol;
}

// ─────────────────────────────────────────────────────────────────────────────
// propagate() — one-step RK4 integration used by the simulation loop
//
// RK4 is more accurate than Euler and avoids numerical drift over long runs.
// The NLP uses Euler for speed, so there is a small model mismatch; the
// closed-loop re-estimation of s after each step corrects for it.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> MPC::propagate(const std::vector<double>& x,
                                    const std::vector<double>& u) const
{
    DM xd(x), ud(u), dtd(dt_);
    DM xn = f_disc_(std::vector<DM>{xd, ud, dtd}).at(0);
    return xn.get_elements();
}
