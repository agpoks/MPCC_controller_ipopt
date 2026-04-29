#pragma once
#include <casadi/casadi.hpp>
#include <vector>
#include <map>
#include <string>

/*
 * MPCC controller — single-track vehicle model with Pacejka tires.
 * Based on Liniger et al. (2017) as implemented in demo/C++/.
 *
 * State  x = [X, Y, phi, vx, vy, r, s, D, delta, vs]       (NX = 10)
 *   X, Y    : global position
 *   phi     : heading angle
 *   vx, vy  : body-frame longitudinal / lateral velocity
 *   r       : yaw rate
 *   s       : arc-length progress along reference track
 *   D       : drivetrain force command  (Frx = Cm1*D - Cm2*D*vx)
 *   delta   : front steering angle
 *   vs      : progress speed  (ds/dt)
 *
 * Input  u = [dD, dDelta, dVs]                              (NU = 3)
 *   Rates of change of the actuator states D, delta, vs.
 *
 * Tyre forces: Pacejka simplified (Liniger model):
 *   Ffy = Df * sin(Cf * atan(Bf * alpha_f))
 *   Fry = Dr * sin(Cr * atan(Br * alpha_r))
 *   Frx = Cm1*D - Cm2*D*vx
 *   Ffric = -Cr0 - Cr2*vx^2
 *
 * Integration: RK4, fixed dt.
 * Solver: CasADi Opti + IPOPT — fully nonlinear, no linearisation.
 * Obstacle avoidance: hard circular constraints (parametric per solve).
 */

static constexpr int NX = 10;
static constexpr int NU = 3;

struct VehicleParams {
    double Cm1 = 0.287,  Cm2 = 0.0545;           // drivetrain
    double Cr0 = 0.0518, Cr2 = 0.00035;           // friction
    double Bf  = 2.579,  Cf = 1.2,   Df = 0.192; // front Pacejka
    double Br  = 3.3852, Cr = 1.2691, Dr = 0.1737;// rear Pacejka
    double m   = 0.041,  Iz = 27.8e-6;            // inertia
    double lf  = 0.029,  lr = 0.033;              // axle distances
};

class MPC {
public:
    struct Obstacle { double x, y, radius; };

    struct Solution {
        casadi::DM X_opt;                // (N+1) x NX
        casadi::DM U_opt;                // N x NU  [dD, dDelta, dVs]
        std::vector<double> u_cmd_first; // first optimal [dD, dDelta, dVs]
    };

    double theta_max       = 0.35; // delta_max exposed for plotter
    double last_cost       = 0.0;
    double last_solve_time = 0.0;

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

    void setup_MPC();
    void set_obstacles(const std::vector<Obstacle>& obs);

    // x0 must have exactly NX=10 elements: [X,Y,phi,vx,vy,r,s,D,delta,vs]
    Solution solve(const std::vector<double>& x0);

    // One-step numerical simulation using the same RK4 dynamics.
    std::vector<double> propagate(const std::vector<double>& x,
                                  const std::vector<double>& u) const;

private:
    int    N_   = 15;
    double dt_  = 0.05;
    int    max_obs_    = 4;
    double obs_margin_ = 0.25;
    double car_radius_ = 0.15;

    double vx_min_ = 0.05, vx_max_ = 3.5;
    double vy_min_ = -3.0, vy_max_ = 3.0;
    double r_min_  = -8.0, r_max_  = 8.0;
    double D_min_  = -0.1, D_max_  = 1.0;
    double delta_min_ = -0.35, delta_max_ = 0.35;
    double vs_min_ = 0.0,  vs_max_ = 3.5;
    double X_min_  = -200.0, X_max_ = 200.0;
    double Y_min_  = -200.0, Y_max_ = 200.0;
    double s_ext_max_ = 300.0;

    double dD_max_     = 15.0;
    double dDelta_max_ = 15.0;
    double dVs_max_    = 10.0;

    double qC_      = 0.1;
    double qL_      = 500.0;
    double qVs_     = 0.02;
    double qVref_   = 0.0;
    double rdD_     = 1e-4;
    double rdDelta_ = 5e-3;
    double rdVs_    = 1e-5;
    double qObs_    = 2000.0;
    double qObsHardSlack_ = 20000.0;
    bool   use_hard_obs_slack_ = true;
    double qCN_mult_ = 10.0;

    int    ipopt_max_iter_       = 100;
    double ipopt_tol_            = 1e-4;
    double ipopt_acceptable_tol_ = 1e-3;
    double ipopt_acceptable_obj_change_tol_ = 1e-3;
    double ipopt_max_cpu_time_   = 0.0;
    std::string ipopt_linear_solver_ = "ma57";  // HSL MA57 (faster than MUMPS)
    std::string ipopt_hsllib_        = "";

    VehicleParams vp_;
    casadi::Function f_cont_;  // continuous dynamics x_dot=f(x,u) — Euler in NLP
    casadi::Function f_disc_;  // RK4 discrete dynamics — used only in propagate()

    casadi::Function c_lut_x_, c_lut_y_, c_lut_dx_, c_lut_dy_;
    casadi::Function r_lut_x_, r_lut_y_, l_lut_x_,  l_lut_y_;
    casadi::Function v_ref_lut_;
    double s_total_ = 0.0;

    casadi::Opti opti_;
    casadi::MX   X_, U_;
    casadi::MX   p_x0_, p_obs_, p_u_prev_;
    // Track geometry parameters: (N+1) rows × 8 cols [cx,cy,tx,ty,lx,ly,rx,ry]
    // Evaluated numerically before each solve at warm-start arc-lengths.
    // Avoids 8*(N+1) symbolic B-spline nodes in the NLP graph.
    casadi::MX   p_track_;
    casadi::MX   p_vref_;
    casadi::MX   cost_expr_;

    casadi::DM X_warm_, U_warm_;
    casadi::DM u_prev_;
    bool       has_warm_start_ = false;

    std::vector<Obstacle> obstacles_;

    void build_dynamics();
    void check_and_fix_boundary_orientation();
    void add_corridor_constraint(const casadi::MX& X, const casadi::MX& Y,
                                 const casadi::MX& s_c);
    void add_obstacle_constraints(const casadi::MX& X, const casadi::MX& Y);
    casadi::MX obstacle_penalty(const casadi::MX& X, const casadi::MX& Y) const;
    void configure_solver();
};
