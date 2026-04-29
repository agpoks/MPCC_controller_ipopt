#include "MPCC_controller.h"

#include <chrono>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <cstring>

using namespace casadi;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static MX lut_mx(const Function& f, const MX& s)
{
    return f(std::vector<MX>{s}).at(0);
}

static double lut_num(const Function& f, double s)
{
    return static_cast<double>(f(std::vector<DM>{DM(s)}).at(0));
}

// ─────────────────────────────────────────────────────────────────────────────
// CasADi single-track dynamics + RK4 discretisation
//
// Matches demo/C++/Model/model.cpp and demo/vehicle_dynamics_models/
// single_track_planar_model, implemented directly in SX for CasADi.
// ─────────────────────────────────────────────────────────────────────────────
void MPC::build_dynamics()
{
    using casadi::SX;

    SX x  = SX::sym("x",  NX);
    SX u  = SX::sym("u",  NU);
    SX dt = SX::sym("dt", 1);

    // ── State extraction ──────────────────────────────────────────────────
    SX phi   = x(2);
    SX vx    = x(3), vy = x(4), r = x(5);
    SX D     = x(7), delta = x(8), vs = x(9);

    // ── Input extraction ──────────────────────────────────────────────────
    SX dD = u(0), dDelta = u(1), dVs = u(2);

    // ── Tyre slip angles (atan form, 1e-3 regularisation on vx) ──────────
    // alpha_f = delta - atan2(vy + lf*r, vx)  [Liniger eq. 6a simplified]
    // alpha_r =       - atan2(vy - lr*r, vx)  [Liniger eq. 6b simplified]
    const double eps = 1e-3;
    SX alpha_f = delta - atan((vp_.lf * r + vy) / (vx + eps));
    SX alpha_r =        -atan((vy - vp_.lr * r) / (vx + eps));

    // ── Pacejka lateral forces (simplified, no E term) ───────────────────
    SX Ffy = vp_.Df * sin(vp_.Cf * atan(vp_.Bf * alpha_f));
    SX Fry = vp_.Dr * sin(vp_.Cr * atan(vp_.Br * alpha_r));

    // ── Longitudinal forces ───────────────────────────────────────────────
    SX Frx   = vp_.Cm1 * D - vp_.Cm2 * D * vx;
    SX Ffric = -vp_.Cr0 - vp_.Cr2 * pow(vx, 2);

    // ── Body-frame velocity dynamics ──────────────────────────────────────
    SX vx_dot = (Frx + Ffric - Ffy * sin(delta) + vp_.m * vy * r) / vp_.m;
    SX vy_dot = (Fry + Ffy * cos(delta) - vp_.m * vx * r) / vp_.m;
    SX r_dot  = (Ffy * vp_.lf * cos(delta) - Fry * vp_.lr) / vp_.Iz;

    // ── Global kinematics ─────────────────────────────────────────────────
    SX X_dot   = vx * cos(phi) - vy * sin(phi);
    SX Y_dot   = vx * sin(phi) + vy * cos(phi);
    SX phi_dot = r;

    // ── Actuator-state dynamics ───────────────────────────────────────────
    SX s_dot     = vs;
    SX D_dot     = dD;
    SX delta_dot = dDelta;
    SX vs_dot    = dVs;

    // vertcat with > 6 args requires the vector overload
    SX f_expr = vertcat(std::vector<SX>{
        X_dot, Y_dot, phi_dot,
        vx_dot, vy_dot, r_dot,
        s_dot, D_dot, delta_dot, vs_dot});

    f_cont_ = Function("f_cont",
                       std::vector<SX>{x, u},
                       std::vector<SX>{f_expr},
                       std::vector<std::string>{"x","u"},
                       std::vector<std::string>{"xdot"});
    // Local alias for RK4 sub-step calls below
    Function f_cont = f_cont_;

    // ── RK4 integration ───────────────────────────────────────────────────
    auto call = [&](const SX& xx, const SX& uu) -> SX {
        return f_cont(std::vector<SX>{xx, uu})[0];
    };
    SX k1 = call(x,                   u);
    SX k2 = call(x + dt * 0.5 * k1,   u);
    SX k3 = call(x + dt * 0.5 * k2,   u);
    SX k4 = call(x + dt       * k3,   u);
    SX xn = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);

    f_disc_ = Function("f_disc",
                       std::vector<SX>{x, u, dt},
                       std::vector<SX>{xn},
                       std::vector<std::string>{"x", "u", "dt"},
                       std::vector<std::string>{"xip1"});
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter loading
// ─────────────────────────────────────────────────────────────────────────────
void MPC::set_initial_params(const std::map<std::string, double>& num,
                              const std::map<std::string, std::string>& /*str*/)
{
    auto get = [&](const std::string& k, double def) {
        auto it = num.find(k);
        return (it != num.end()) ? it->second : def;
    };

    N_   = static_cast<int>(get("N",  20));
    dt_  = get("dT", 0.05);

    delta_max_ = get("theta_max", 0.35);
    theta_max  = delta_max_;

    vx_max_  = get("v_max",    3.5);
    vs_max_  = get("p_max",    vx_max_);
    vs_min_  = get("p_min",    0.0);
    s_ext_max_ = get("s_max", 300.0);

    max_obs_    = static_cast<int>(get("max_obstacles", 4));
    obs_margin_ = get("obs_margin", 0.25);

    qC_       = get("mpc_w_cte",     0.1);
    qL_       = get("mpc_w_lag",   500.0);
    qVs_      = get("mpc_w_p",      0.02);
    qVref_    = get("mpc_w_vref",    2.0);
    rdD_      = get("mpc_w_accel",   1e-4);
    rdDelta_  = get("mpc_w_delta_d", 5e-3);
    rdVs_     = get("mpc_w_delta_p", 1e-5);
    qObs_     = get("mpc_w_obs", 2000.0);
    qObsHardSlack_ = get("mpc_w_obs_hard_slack", 20000.0);
    use_hard_obs_slack_ = (get("mpc_use_hard_obs_slack", 1.0) > 0.5);

    ipopt_max_iter_       = static_cast<int>(get("ipopt_max_iter", 100));
    ipopt_tol_            = get("ipopt_tol",            1e-4);
    ipopt_acceptable_tol_ = get("ipopt_acceptable_tol", 1e-3);
    ipopt_acceptable_obj_change_tol_ = get("ipopt_acceptable_obj_change_tol", 1e-3);
    ipopt_max_cpu_time_   = get("ipopt_max_cpu_time", 0.0);

    // ── HSL linear solver (environment variable overrides) ──────────────
    const char* env_solver = std::getenv("MPC_IPOPT_LINEAR_SOLVER");
    if (env_solver && std::string(env_solver).size() > 0) {
        ipopt_linear_solver_ = std::string(env_solver);
    }
    const char* env_hsllib = std::getenv("MPC_IPOPT_HSL_LIB");
    if (env_hsllib && std::string(env_hsllib).size() > 0) {
        ipopt_hsllib_ = std::string(env_hsllib);
    } else {
        const char* env_hsllib_alt1 = std::getenv("IPOPT_HSL_LIB");
        const char* env_hsllib_alt2 = std::getenv("HSL_LIB");
        if (env_hsllib_alt1 && std::string(env_hsllib_alt1).size() > 0) {
            ipopt_hsllib_ = std::string(env_hsllib_alt1);
        } else if (env_hsllib_alt2 && std::string(env_hsllib_alt2).size() > 0) {
            ipopt_hsllib_ = std::string(env_hsllib_alt2);
        }
    }

    // If no env var is set, try common local ThirdParty-HSL locations.
    if (ipopt_hsllib_.empty()) {
        namespace fs = std::filesystem;
        const char* home = std::getenv("HOME");
        if (home && std::string(home).size() > 0) {
            const fs::path cand1 = fs::path(home) / "ThirdParty-HSL/.libs/libcoinhsl.so";
            const fs::path cand2 = fs::path(home) / "ThirdParty-HSL/.libs/libhsl.so";
            if (fs::exists(cand1)) {
                ipopt_hsllib_ = cand1.string();
            } else if (fs::exists(cand2)) {
                ipopt_hsllib_ = cand2.string();
            }
        }
    }

    // Some CasADi/IPOPT builds do not expose the 'hsllib' IPOPT option.
    // Make HSL visible by extending LD_LIBRARY_PATH with the library folder.
    if (!ipopt_hsllib_.empty()) {
        namespace fs = std::filesystem;
        fs::path libp(ipopt_hsllib_);
        if (fs::exists(libp)) {
            const std::string libdir = libp.parent_path().string();
            const char* ld = std::getenv("LD_LIBRARY_PATH");
            std::string cur = ld ? std::string(ld) : std::string();
            if (cur.find(libdir) == std::string::npos) {
                std::string upd = libdir;
                if (!cur.empty()) upd += ":" + cur;
                setenv("LD_LIBRARY_PATH", upd.c_str(), 1);
            }
        }
    }

    // Initial previous input: zero rates → start from rest
    u_prev_ = DM::zeros(NU);
}

// ─────────────────────────────────────────────────────────────────────────────
// Track data
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
// Corridor orientation check — ensure left boundary has positive e_c offset.
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
        // n = [-ty, tx] (left-pointing normal); e_c_l = n^T*(l-c)
        double e_c_l = -ty*(lx - cx) + tx*(ly - cy);
        if (e_c_l < 0) vote++;
    }
    if (vote > N_CHECK / 2) {
        std::swap(l_lut_x_, r_lut_x_);
        std::swap(l_lut_y_, r_lut_y_);
        std::cout << "[MPCC] Auto-corrected left/right boundary orientation.\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Corridor constraint using numeric track refs (from p_track_ parameter row k)
// Row k of p_track_: [cx, cy, tx, ty, lx, ly, rx, ry]
// ─────────────────────────────────────────────────────────────────────────────
void MPC::add_corridor_constraint(const MX& Xk, const MX& Yk, const MX& s_c)
{
    // This overload is now unused — corridor is inlined in setup_MPC using
    // p_track_ rows for numerical stability. Keep stub for linker.
    (void)Xk; (void)Yk; (void)s_c;
}

// ─────────────────────────────────────────────────────────────────────────────
// Obstacle avoidance — soft quadratic penalty in cost (always feasible).
// Returns the obstacle penalty term to be added to the stage cost.
// ─────────────────────────────────────────────────────────────────────────────
MX MPC::obstacle_penalty(const MX& Xk, const MX& Yk) const
{
    const double min_sep = car_radius_ + obs_margin_;
    MX pen = MX::zeros(1);
    for (int i = 0; i < max_obs_; i++) {
        MX ox  = p_obs_(i, 0), oy  = p_obs_(i, 1), or_ = p_obs_(i, 2);
        MX r_req_sq = pow(or_ + min_sep, 2);
        MX dist_sq  = pow(Xk - ox, 2) + pow(Yk - oy, 2);
        // Penalty = max(0, r_req^2 - dist^2)^2 / r_req^2  (normalised)
        MX excess   = r_req_sq - dist_sq;          // > 0 when too close
        MX pos_exc  = (excess + fabs(excess)) / 2; // soft ReLU  (non-negative part)
        pen += qObs_ * pow(pos_exc, 2) / (r_req_sq + 1e-6);
    }
    return pen;
}

// legacy stub (unused — kept so linker doesn't error if called)
void MPC::add_obstacle_constraints(const MX& Xk, const MX& Yk)
{
    (void)Xk; (void)Yk;
}

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
    ipopt_opts["warm_start_mult_bound_push"] = 1e-6;
    ipopt_opts["warm_start_slack_bound_push"] = 1e-6;
    ipopt_opts["mu_init"]                   = 1e-3;
    ipopt_opts["mu_strategy"]               = std::string("adaptive");
    ipopt_opts["nlp_scaling_method"]        = std::string("gradient-based");
    ipopt_opts["fixed_variable_treatment"]  = std::string("relax_bounds");
    ipopt_opts["linear_solver"]             = ipopt_linear_solver_;
    if (ipopt_max_cpu_time_ > 0.0) {
        ipopt_opts["max_cpu_time"]          = ipopt_max_cpu_time_;
    }

    Dict solver_opts;
    solver_opts["ipopt"]      = ipopt_opts;
    solver_opts["verbose"]    = false;
    solver_opts["print_time"] = false;
    solver_opts["expand"]     = true;

    opti_.solver("ipopt", solver_opts);
}

// ─────────────────────────────────────────────────────────────────────────────
// Build NLP (called once before the simulation loop)
// ─────────────────────────────────────────────────────────────────────────────
void MPC::setup_MPC()
{
    check_and_fix_boundary_orientation();
    build_dynamics();   // creates f_disc_

    opti_ = Opti();

    // Decision variables — each column = one time step
    X_        = opti_.variable(NX, N_ + 1);
    U_        = opti_.variable(NU, N_);
    // Soft corridor slack per future state (k+1 .. N): always keeps NLP feasible.
    // The vehicle pays a large quadratic cost for violating the corridor.
    MX S_cor  = opti_.variable(1, N_);  // one slack per step, >= 0
    // Optional hard obstacle-distance constraints with non-negative slack.
    // This keeps feasibility while strongly discouraging entering obstacle discs.
    MX S_obs;
    if (use_hard_obs_slack_) {
        S_obs = opti_.variable(max_obs_, N_);
    }

    p_x0_     = opti_.parameter(NX);
    p_obs_    = opti_.parameter(max_obs_, 3);
    p_u_prev_ = opti_.parameter(NU);
    p_vref_   = opti_.parameter(N_ + 1);
    // Track geometry at each horizon step (rows = steps 0..N, cols = 8 values).
    // Evaluated numerically before each solve — avoids symbolic LUT nodes in NLP.
    // Columns: [cx, cy, tx, ty, lx, ly, rx, ry]
    p_track_  = opti_.parameter(N_ + 1, 8);

    opti_.subject_to(S_cor >= 0.0);    // slack must be non-negative
    if (use_hard_obs_slack_) {
        opti_.subject_to(opti_.bounded(0.0, S_obs, std::numeric_limits<double>::infinity()));
    }

    // Initial state constraint
    opti_.subject_to(X_(Slice(), 0) == p_x0_);

    MX total_cost = MX::zeros(1);

    for (int k = 0; k < N_; k++) {
        MX xk = X_(Slice(), k);   // NX x 1
        MX uk = U_(Slice(), k);   // NU x 1
        MX xkp1 = X_(Slice(), k+1);

        MX Xk = xk(0), Yk = xk(1), vsk = xk(9);
        MX vrefk = p_vref_(k);

        // ── Track geometry from numeric parameter (avoids symbolic LUT graph) ─
        MX cx = p_track_(k, 0), cy = p_track_(k, 1);
        MX tx = p_track_(k, 2), ty = p_track_(k, 3);
        MX lx = p_track_(k, 4), ly = p_track_(k, 5);
        MX rx = p_track_(k, 6), ry = p_track_(k, 7);

        // ── MPCC errors ────────────────────────────────────────────────
        MX epX = Xk - cx, epY = Yk - cy;
        MX e_l = tx*epX + ty*epY;           // lag error
        MX e_c = -ty*epX + tx*epY;          // contouring error (+ = left of track)

        // Corridor bounds (used in cost and next-state constraints below)
        MX e_c_l = -ty*(lx-cx) + tx*(ly-cy);   // left offset (> 0)
        MX e_c_r = -ty*(rx-cx) + tx*(ry-cy);   // right offset (< 0)

        // ── Stage cost ─────────────────────────────────────────────────
        MX uk_prev = (k == 0) ? MX(p_u_prev_) : MX(U_(Slice(), k-1));
        MX du = uk - uk_prev;

        total_cost += qC_    * pow(e_c, 2)
                   +  qL_    * pow(e_l, 2)
                   -  qVs_   * vsk
                   +  qVref_ * pow(vsk - vrefk, 2)
                   +  rdD_   * pow(du(0), 2)
                   +  rdDelta_* pow(du(1), 2)
               +  rdVs_  * pow(du(2), 2);

        // ── Dynamics: Euler step (quarter the expression graph of RK4) ──
        // propagate() uses RK4 for accuracy; NLP uses Euler for speed.
        MX xdot = f_cont_(std::vector<MX>{xk, uk}).at(0);
        opti_.subject_to(xkp1 == xk + dt_ * xdot);

        // ── Control bounds ─────────────────────────────────────────────
        opti_.subject_to(uk(0) >= -dD_max_);     opti_.subject_to(uk(0) <= dD_max_);
        opti_.subject_to(uk(1) >= -dDelta_max_); opti_.subject_to(uk(1) <= dDelta_max_);
        opti_.subject_to(uk(2) >= -dVs_max_);    opti_.subject_to(uk(2) <= dVs_max_);

        // ── State constraints on NEXT state ────────────────────────────
        MX X1  = xkp1(0), Y1 = xkp1(1), vx1 = xkp1(3);
        MX D1  = xkp1(7), d1 = xkp1(8), vs1 = xkp1(9), s1 = xkp1(6);

        // Only constrain actuator states (direct physical limits, always satisfiable)
        // and arc-length. Velocity states (vx, vy, r) are NOT hard-bounded here because
        // Euler dynamics + hard bounds cause joint infeasibility when vx ≈ vx_max.
        // Speed envelope is enforced via clamp in the simulation loop (main.cpp).
        opti_.subject_to(D1  >= D_min_);     opti_.subject_to(D1  <= D_max_);
        opti_.subject_to(d1  >= delta_min_); opti_.subject_to(d1  <= delta_max_);
        opti_.subject_to(vs1 >= vs_min_);    opti_.subject_to(vs1 <= vs_max_);
        opti_.subject_to(s1  >= 0.0);        opti_.subject_to(s1  <= s_ext_max_);

        // Progress speed coupling: prevents IPOPT from pushing vs > vehicle speed.
        // Use fmax(0, vx1) so the constraint stays feasible even if vx briefly < 0
        // (Euler integration overshoot in hard braking scenarios).
        opti_.subject_to(vs1 <= fmax(MX(0.0), vx1) + 0.5);

        // Soft obstacle penalty on next state
        total_cost += obstacle_penalty(X1, Y1);

        // Optional near-hard obstacle avoidance on next state:
        // dist^2 + slack >= (r_obs + car_radius + margin)^2
        if (use_hard_obs_slack_) {
            const double min_sep = car_radius_ + obs_margin_;
            for (int i = 0; i < max_obs_; ++i) {
                MX ox = p_obs_(i, 0), oy = p_obs_(i, 1), or_ = p_obs_(i, 2);
                MX req_sq = pow(or_ + min_sep, 2);
                MX dist_sq = pow(X1 - ox, 2) + pow(Y1 - oy, 2);
                MX sk_obs = S_obs(i, k);
                opti_.subject_to(opti_.bounded(0.0,
                                               dist_sq + sk_obs - req_sq,
                                               std::numeric_limits<double>::infinity()));
                total_cost += qObsHardSlack_ * pow(sk_obs, 2);
            }
        }

        // Soft corridor on next state: add slack S_cor(k) to the cost
        {
            MX cx1 = p_track_(k+1, 0), cy1 = p_track_(k+1, 1);
            MX tx1 = p_track_(k+1, 2), ty1 = p_track_(k+1, 3);
            MX lx1 = p_track_(k+1, 4), ly1 = p_track_(k+1, 5);
            MX rx1 = p_track_(k+1, 6), ry1 = p_track_(k+1, 7);
            MX e_c1   = -ty1*(X1-cx1) + tx1*(Y1-cy1);
            MX e_c_l1 = -ty1*(lx1-cx1) + tx1*(ly1-cy1);
            MX e_c_r1 = -ty1*(rx1-cx1) + tx1*(ry1-cy1);
            // Hard corridor with slack: e_c_r - sk <= e_c1 <= e_c_l + sk
            MX sk = S_cor(k);
            opti_.subject_to(e_c1 <= e_c_l1 + sk);
            opti_.subject_to(e_c1 >= e_c_r1 - sk);
            // Large penalty on corridor slack
            const double q_s = 500.0;
            total_cost += q_s * pow(sk, 2);
        }
    }

    // ── Terminal cost using last row of p_track_ ─────────────────────────
    {
        MX xN = X_(Slice(), N_);
        MX XN = xN(0), YN = xN(1);
        MX cx = p_track_(N_, 0), cy = p_track_(N_, 1);
        MX tx = p_track_(N_, 2), ty = p_track_(N_, 3);
        MX epX = XN - cx, epY = YN - cy;
        MX e_l_N = tx*epX + ty*epY;
        MX e_c_N = -ty*epX + tx*epY;
        total_cost += qCN_mult_ * qC_ * pow(e_c_N, 2)
                    + qCN_mult_ * qL_ * pow(e_l_N, 2);
    }

    opti_.minimize(total_cost);
    cost_expr_ = total_cost;

    configure_solver();
    std::cout << "[MPCC] NLP built: N=" << N_ << " dt=" << dt_
              << " s, nx=" << NX << " nu=" << NU
              << " max_obs=" << max_obs_
              << " linear_solver=" << ipopt_linear_solver_
              << " hsl_lib_path=" << (ipopt_hsllib_.empty() ? std::string("<none>") : ipopt_hsllib_)
              << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Set obstacles for next solve
// ─────────────────────────────────────────────────────────────────────────────
void MPC::set_obstacles(const std::vector<Obstacle>& obs)
{
    obstacles_ = obs;
}

// ─────────────────────────────────────────────────────────────────────────────
// Solve MPCC at current state x0 (NX elements)
// ─────────────────────────────────────────────────────────────────────────────
MPC::Solution MPC::solve(const std::vector<double>& x0)
{
    // ── Obstacle matrix (pad inactive entries far away) ───────────────────
    DM obs_mat = DM::zeros(max_obs_, 3);
    for (int i = 0; i < max_obs_; i++) {
        if (i < static_cast<int>(obstacles_.size())) {
            obs_mat(i, 0) = obstacles_[i].x;
            obs_mat(i, 1) = obstacles_[i].y;
            obs_mat(i, 2) = obstacles_[i].radius;
        } else {
            obs_mat(i, 0) = 1e4;
            obs_mat(i, 1) = 1e4;
            obs_mat(i, 2) = 0.0;
        }
    }

    opti_.set_value(p_x0_,    DM(x0));
    opti_.set_value(p_obs_,   obs_mat);
    opti_.set_value(p_u_prev_, u_prev_);

    // ── Populate track geometry parameter numerically ─────────────────────
    // Always based on current x0[6] (real arc-length) and estimated vs.
    // This ensures the geometry stays correct even after failed solves.
    {
        DM track_vals = DM::zeros(N_ + 1, 8);
        DM vref_vals  = DM::zeros(N_ + 1);
        double s0 = x0[6];
        double vs_est = std::clamp(x0[9], vs_min_, std::min(x0[3] + 0.5, vs_max_));

        for (int k = 0; k <= N_; k++) {
            double sk = std::min(s0 + k * dt_ * vs_est, s_ext_max_);
            track_vals(k, 0) = lut_num(c_lut_x_,  sk);
            track_vals(k, 1) = lut_num(c_lut_y_,  sk);
            track_vals(k, 2) = lut_num(c_lut_dx_, sk);
            track_vals(k, 3) = lut_num(c_lut_dy_, sk);
            track_vals(k, 4) = lut_num(l_lut_x_,  sk);
            track_vals(k, 5) = lut_num(l_lut_y_,  sk);
            track_vals(k, 6) = lut_num(r_lut_x_,  sk);
            track_vals(k, 7) = lut_num(r_lut_y_,  sk);
            vref_vals(k)     = v_ref_lut_.is_null() ? vx_max_ : lut_num(v_ref_lut_, sk);
        }
        opti_.set_value(p_track_, track_vals);
        opti_.set_value(p_vref_,  vref_vals);
    }

    // ── Build cold start (centerline-following, equilibrium throttle) ────────
    auto make_cold_start = [&]() -> std::pair<DM, DM> {
        double s0  = x0[6];
        double vx0 = std::max(vx_min_, x0[3]);
        // Equilibrium throttle: Cm1*D - Cm2*D*vx = Cr0 + Cr2*vx^2 => D = ...
        double denom = vp_.Cm1 - vp_.Cm2 * vx0;
        double D_eq  = std::clamp(
            (denom > 1e-6) ? (vp_.Cr0 + vp_.Cr2 * vx0 * vx0) / denom : 0.0,
            D_min_, D_max_);
        double vs0 = std::clamp(x0[9], vs_min_, std::min(vx0 + 0.5, vs_max_));

        DM Xi = DM::zeros(NX, N_ + 1);
        DM Ui = DM::zeros(NU, N_);
        for (int j = 0; j < NX; j++) Xi(j, 0) = x0[j];

        for (int k = 1; k <= N_; k++) {
            double sk = std::min(s0 + k * dt_ * vs0, s_ext_max_);
            Xi(0, k) = lut_num(c_lut_x_,  sk);
            Xi(1, k) = lut_num(c_lut_y_,  sk);
            Xi(2, k) = std::atan2(lut_num(c_lut_dy_, sk), lut_num(c_lut_dx_, sk));
            Xi(3, k) = vx0;
            Xi(4, k) = 0.0;
            Xi(5, k) = 0.0;
            Xi(6, k) = sk;
            Xi(7, k) = D_eq;
            Xi(8, k) = 0.0;
            Xi(9, k) = vs0;
        }
        return {Xi, Ui};
    };

    // ── Set initial guess ─────────────────────────────────────────────────
    if (has_warm_start_) {
        opti_.set_initial(X_, X_warm_);
        opti_.set_initial(U_, U_warm_);
    } else {
        auto [Xi, Ui] = make_cold_start();
        opti_.set_initial(X_, Xi);
        opti_.set_initial(U_, Ui);
    }

    // ── Solve ─────────────────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = true;
    try {
        opti_.solve();
    } catch (const std::exception& e) {
        ok = false;
        const std::string msg = e.what();
        std::cerr << "[MPCC] " << msg << "\n";

        // If IPOPT rejects a configured option (commonly unsupported linear solver),
        // switch to MUMPS once and retry immediately.
        if (msg.find("Invalid_Option") != std::string::npos && ipopt_linear_solver_ != "mumps") {
            std::cerr << "[MPCC] Falling back IPOPT linear solver from '" << ipopt_linear_solver_
                      << "' to 'mumps'.\n";
            ipopt_linear_solver_ = "mumps";
            configure_solver();
            try {
                opti_.solve();
                ok = true;
            } catch (const std::exception& e2) {
                std::cerr << "[MPCC] Retry with mumps failed: " << e2.what() << "\n";
            }
        }
    }
    last_solve_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // ── Extract solution ──────────────────────────────────────────────────
    // When IPOPT fails, the debug iterate may be far outside bounds and cause
    // the dynamic model to blow up.  Safe fallback priority:
    //   1. Solved ok         → use optimal values, update warm start
    //   2. Failed + warm start available → keep previous warm start, zero rates
    //   3. Failed, no history → apply cold-start guess (very conservative)
    Solution sol;
    sol.u_cmd_first.assign(NU, 0.0);

    if (ok) {
        DM Xsol = opti_.value(X_);
        DM Usol = opti_.value(U_);
        sol.X_opt = Xsol.T();
        sol.U_opt = Usol.T();
        try { last_cost = static_cast<double>(opti_.value(cost_expr_)); }
        catch (...) { last_cost = 0.0; }
        // Extract first input
        for (int j = 0; j < NU; j++)
            sol.u_cmd_first[j] = static_cast<double>(Usol(j, 0));
        // Update warm start: shift by one step
        X_warm_ = horzcat(Xsol(Slice(), Slice(1, N_+1)),
                          Xsol(Slice(), Slice(N_, N_+1)));
        U_warm_ = horzcat(Usol(Slice(), Slice(1, N_)),
                          Usol(Slice(), Slice(N_-1, N_)));
        has_warm_start_ = true;
    } else if (has_warm_start_) {
        // Use shifted warm start and apply first warm-start control.
        // This is safer than zero rates when a real-time solve times out.
        sol.X_opt = X_warm_.T();
        sol.U_opt = U_warm_.T();
        last_cost = 0.0;
        if (U_warm_.size2() > 0) {
            sol.u_cmd_first[0] = static_cast<double>(U_warm_(0, 0));
            sol.u_cmd_first[1] = static_cast<double>(U_warm_(1, 0));
            sol.u_cmd_first[2] = static_cast<double>(U_warm_(2, 0));
        } else {
            sol.u_cmd_first = {0.0, 0.0, 0.0};
        }
        // Don't update warm start (keep previous feasible trajectory)
    } else {
        // No history at all: cold start output — zero rates
        auto [Xi, Ui] = make_cold_start();
        sol.X_opt = Xi.T();
        sol.U_opt = Ui.T();
        last_cost = 0.0;
        sol.u_cmd_first = {0.0, 0.0, 0.0};
    }

    // ── Always clamp first control to physical bounds ─────────────────────
    sol.u_cmd_first[0] = std::clamp(sol.u_cmd_first[0], -dD_max_,     dD_max_);
    sol.u_cmd_first[1] = std::clamp(sol.u_cmd_first[1], -dDelta_max_, dDelta_max_);
    sol.u_cmd_first[2] = std::clamp(sol.u_cmd_first[2], -dVs_max_,    dVs_max_);

    // Update previous input for rate cost at next step
    u_prev_ = DM({sol.u_cmd_first[0], sol.u_cmd_first[1], sol.u_cmd_first[2]});

    return sol;
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerical one-step propagation (for the simulation loop in main.cpp)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> MPC::propagate(const std::vector<double>& x,
                                    const std::vector<double>& u) const
{
    DM xd(x), ud(u), dtd(dt_);
    DM xn = f_disc_(std::vector<DM>{xd, ud, dtd}).at(0);
    return xn.get_elements();
}
