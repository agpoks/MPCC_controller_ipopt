#pragma once
#include <casadi/casadi.hpp>
#include <vector>
#include <map>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// MPCC (Model Predictive Contouring Control) — single-track vehicle
//
// Reference: Liniger et al., "Optimization-Based Autonomous Racing of 1:43
// Scale Vehicles", Optimal Control Applications and Methods, 2017.
//
// Vehicle model: Euler-discretised single-track with Pacejka simplified tyre
// forces (no combined-slip), full CasADi / IPOPT NLP.  HSL MA57 linear solver.
//
// ── State  x ∈ ℝ^10 ──────────────────────────────────────────────────────
//   idx  sym   meaning
//    0    X    global x-position [m]
//    1    Y    global y-position [m]
//    2    φ    heading angle [rad]
//    3    vx   body-frame longitudinal velocity [m/s]
//    4    vy   body-frame lateral velocity [m/s]
//    5    r    yaw rate [rad/s]
//    6    s    arc-length progress along reference track [m]
//    7    D    normalised drivetrain command  (Frx = Cm1·D − Cm2·D·vx)
//    8    δ    front steering angle [rad]
//    9    vs   virtual progress speed  ds/dt [m/s]
//
// ── Input  u ∈ ℝ^3 ───────────────────────────────────────────────────────
//   idx  sym    meaning
//    0   dD     rate of change of D  [1/s]
//    1   dδ     rate of change of δ  [rad/s]
//    2   dvs    rate of change of vs [m/s²]
//
// ── Tyre forces (Pacejka simplified) ─────────────────────────────────────
//   α_f = δ − atan2(vy + lf·r, vx)       front slip angle
//   α_r =    − atan2(vy − lr·r, vx)       rear  slip angle
//   Ffy = Df·sin(Cf·atan(Bf·α_f))         front lateral force
//   Fry = Dr·sin(Cr·atan(Br·α_r))         rear  lateral force
//   Frx = Cm1·D − Cm2·D·vx               longitudinal traction
//   Ffric = −Cr0 − Cr2·vx²               rolling resistance
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int NX = 10;
static constexpr int NU = 3;

// Physical parameters of the vehicle (configurable from params/vehicle.yaml).
struct VehicleParams {
    double Cm1 = 0.287,  Cm2 = 0.0545;            // drivetrain
    double Cr0 = 0.0518, Cr2 = 0.00035;            // rolling resistance
    double Bf  = 2.579,  Cf = 1.2,   Df = 0.192;  // front Pacejka
    double Br  = 3.3852, Cr = 1.2691, Dr = 0.1737; // rear  Pacejka
    double m   = 0.041,  Iz = 27.8e-6;             // inertia
    double lf  = 0.029,  lr = 0.033;               // axle distances
};

class MPC {
public:
    struct Obstacle { double x, y, radius; };

    struct Solution {
        casadi::DM X_opt;                // (N+1) × NX trajectory (rows = time steps)
        casadi::DM U_opt;                // N × NU  input sequence
        std::vector<double> u_cmd_first; // first optimal input [dD, dδ, dvs]
    };

    // Exposed so the plotter can read them after each solve.
    double theta_max       = 0.35; // δ_max [rad]
    double last_cost       = 0.0;
    double last_solve_time = 0.0;  // wall-clock time of last IPOPT call [s]

    // Called once before setup_MPC() to set all tuning/solver parameters.
    // Numeric values read from params/mpcc_tuning.yaml (plus vp_ from vehicle.yaml).
    void set_initial_params(const std::map<std::string, double>& num,
                            const std::map<std::string, std::string>& str);

    void set_track_data(const casadi::Function& c_lut_x,
                        const casadi::Function& c_lut_y,
                        const casadi::Function& c_lut_dx,
                        const casadi::Function& c_lut_dy,
                        const casadi::Function& r_lut_x,
                        const casadi::Function& r_lut_y,
                        const casadi::Function& l_lut_x,
                        const casadi::Function& l_lut_y,
                        const std::vector<double>& s_ext,
                        double s_total,
                        const casadi::Function& v_ref_lut);

    // Build the NLP symbolic structure (called once, expensive).
    void setup_MPC();

    // Update obstacle list before each solve call.
    void set_obstacles(const std::vector<Obstacle>& obs);

    // Solve one MPC step.  x0 must have exactly NX elements.
    Solution solve(const std::vector<double>& x0);

    // One-step simulation using RK4 (same dynamics as the NLP, higher accuracy).
    std::vector<double> propagate(const std::vector<double>& x,
                                  const std::vector<double>& u) const;

private:
    // ── Horizon / discretisation ──────────────────────────────────────────
    int    N_   = 15;
    double dt_  = 0.05;

    // ── Physical / track bounds ───────────────────────────────────────────
    double vx_min_ = 0.05, vx_max_ = 3.5;
    double D_min_  = -0.1, D_max_  = 1.0;
    double delta_min_ = -0.35, delta_max_ = 0.35;
    double vs_min_ = 0.0,  vs_max_ = 3.5;
    double s_ext_max_ = 300.0;
    double car_radius_ = 0.15;

    // ── Actuator rate limits ──────────────────────────────────────────────
    double dD_max_     = 15.0;
    double dDelta_max_ = 15.0;
    double dVs_max_    = 10.0;

    // ── Obstacle avoidance ────────────────────────────────────────────────
    int    max_obs_    = 4;
    double obs_margin_ = 0.27;

    // ── MPCC cost weights ─────────────────────────────────────────────────
    double qC_      = 1.0;     // contouring error weight
    double qL_      = 500.0;   // lag error weight
    double qVs_     = 0.05;    // progress reward
    double qVref_   = 3.0;     // speed reference tracking
    double rdD_     = 1e-4;    // throttle-rate penalty
    double rdDelta_ = 0.01;    // steering-rate penalty
    double rdVs_    = 1e-5;    // progress-rate penalty
    double qObs_    = 3000.0;  // soft obstacle penalty
    double qObsHardSlack_ = 20000.0;
    bool   use_hard_obs_slack_ = true;
    double qCN_mult_ = 10.0;   // terminal cost multiplier

    // ── Friction / slip soft constraints ─────────────────────────────────
    double qSlip_              = 50.0;  // penalty per (m/s)² of |vy| above vy_soft_
    double vy_soft_            = 0.25;  // soft lateral-velocity limit [m/s]
    double qFriction_          = 30.0;  // friction-ellipse excess penalty weight
    double friction_long_peak_ = 0.0;   // longitudinal peak force [N]; 0 = auto (vp_.Cm1)
    double friction_lat_peak_  = 0.0;   // lateral peak force [N];      0 = auto (vp_.Dr)

    // ── IPOPT settings ────────────────────────────────────────────────────
    int    ipopt_max_iter_       = 120;
    double ipopt_tol_            = 1e-4;
    double ipopt_acceptable_tol_ = 1e-3;
    double ipopt_acceptable_obj_change_tol_ = 1e-3;
    double ipopt_max_cpu_time_   = 0.0;  // 0 = unlimited (simulation mode)
    std::string ipopt_linear_solver_ = "ma57";
    std::string ipopt_hsllib_        = "";

    // ── Vehicle model ─────────────────────────────────────────────────────
    VehicleParams vp_;
    casadi::Function f_cont_; // continuous dynamics x_dot = f(x,u), used in NLP (Euler)
    casadi::Function f_disc_; // RK4 discrete dynamics, used in propagate()

    // ── Track geometry LUTs ───────────────────────────────────────────────
    casadi::Function c_lut_x_, c_lut_y_, c_lut_dx_, c_lut_dy_;
    casadi::Function r_lut_x_, r_lut_y_, l_lut_x_,  l_lut_y_;
    casadi::Function v_ref_lut_;
    double s_total_ = 0.0;

    // ── CasADi Opti NLP ───────────────────────────────────────────────────
    casadi::Opti opti_;
    casadi::MX   X_, U_;          // decision variables
    casadi::MX   p_x0_;           // initial state parameter (NX × 1)
    casadi::MX   p_obs_;          // obstacle matrix parameter (max_obs × 3)
    casadi::MX   p_u_prev_;       // previous input for rate cost (NU × 1)
    casadi::MX   p_track_;        // numeric track geometry ((N+1) × 8)
    casadi::MX   p_vref_;         // reference speed at each horizon step (N+1)
    casadi::MX   cost_expr_;      // total cost expression (for diagnostics)

    // ── Warm start ────────────────────────────────────────────────────────
    casadi::DM X_warm_, U_warm_;  // shifted solution from previous step
    casadi::DM u_prev_;           // previous applied input (for rate penalty)
    bool       has_warm_start_ = false;

    std::vector<Obstacle> obstacles_;

    void build_dynamics();
    void check_and_fix_boundary_orientation();
    casadi::MX obstacle_penalty(const casadi::MX& X, const casadi::MX& Y) const;
    void configure_solver();
};
