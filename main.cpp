#include "main.h"
#include "MPCC_controller.h"
#include "rl_manager.h"
#include "utils.h"
#include "plotter.h"

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

int main()
{
    // ── Switches ──────────────────────────────────────────────────────────
    bool USE_OBSTACLES = true;

    // ── Parameters ────────────────────────────────────────────────────────
    std::map<std::string, double> param_num;
    std::map<std::string, std::string> param_str;

    // Solver / horizon
    param_num["dT"]  = 0.05;   // sample time [s]
    param_num["N"]   = 15;     // horizon steps (0.75 s horizon)

    // Steering
    param_num["theta_max"] = 0.35;  // max steering angle [rad]

    // Speed / progress bounds
    param_num["v_max"]  = 3.5;
    param_num["v_min"]  = 0.0;
    param_num["p_min"]  = 0.0;   // vs_min
    param_num["p_max"]  = 3.5;   // vs_max

    // Workspace bounds
    param_num["x_min"] = -200.0;  param_num["x_max"] = 200.0;
    param_num["y_min"] = -200.0;  param_num["y_max"] = 200.0;

    // MPCC cost weights (mapped from Liniger names)
    //   mpc_w_cte   → qC   contouring error weight
    //   mpc_w_lag   → qL   lag error weight
    //   mpc_w_p     → qVs  progress maximisation weight
    //   mpc_w_delta_d → rdDelta  steering-rate penalty
    //   mpc_w_delta_p → rdVs     progress-rate penalty
    //   mpc_w_accel   → rdD      throttle-rate penalty
    param_num["mpc_w_cte"]     = 0.1;
    param_num["mpc_w_lag"]     = 500.0;
    param_num["mpc_w_p"]       = 0.02;
    param_num["mpc_w_delta_d"] = 5e-3;
    param_num["mpc_w_delta_p"] = 1e-5;
    param_num["mpc_w_accel"]   = 1e-4;

    // Obstacle settings
    param_num["max_obstacles"] = 4;
    param_num["obs_margin"]    = 0.25;

    // IPOPT
    param_num["ipopt_max_iter"]       = 100;
    param_num["ipopt_tol"]            = 1e-4;
    param_num["ipopt_acceptable_tol"] = 1e-3;

    // ── Track loading ──────────────────────────────────────────────────────
    fs::path project_dir  = fs::path(__FILE__).parent_path();
    fs::path track_folder = project_dir / "raceline";
    std::cout << "Track folder: " << track_folder << "\n";
    if (!fs::exists(track_folder))
        throw std::runtime_error("Track folder not found: " + track_folder.string());

    RLManager rl(track_folder.string());
    TrackData td = rl.load();
    rl.build_reference_speed(1.5, param_num["v_max"], 95.0, "linear");
    td = rl.td();
    param_num["s_max"] = td.s_ext.back();

    // Centerline waypoints + cumulative arc-lengths (for s re-estimation)
    auto P = td.center;
    std::vector<double> S(P.size(), 0.0);
    for (size_t i = 1; i < P.size(); ++i) {
        double dx = P[i][0] - P[i-1][0], dy = P[i][1] - P[i-1][1];
        S[i] = S[i-1] + std::sqrt(dx*dx + dy*dy);
    }
    if (!S.empty()) S.back() = td.s_total;

    // ── MPCC setup ────────────────────────────────────────────────────────
    MPC mpc;
    mpc.set_initial_params(param_num, param_str);
    mpc.set_track_data(
        td.c_lut_x, td.c_lut_y, td.c_lut_dx, td.c_lut_dy,
        td.r_lut_x, td.r_lut_y, td.l_lut_x,  td.l_lut_y,
        td.s_ext, td.s_total, td.v_ref_lut);
    mpc.setup_MPC();

    // ── Initial vehicle state  [X, Y, phi, vx, vy, r, s, D, delta, vs] ──
    double vx0 = 0.3;  // small initial speed to avoid tyre-model singularity
    double phi0 = std::atan2(
        static_cast<double>(td.c_lut_dy(std::vector<casadi::DM>{casadi::DM(0.0)})[0]),
        static_cast<double>(td.c_lut_dx(std::vector<casadi::DM>{casadi::DM(0.0)})[0]));

    std::vector<double> x(NX, 0.0);
    x[0] = td.center[0][0];  // X
    x[1] = td.center[0][1];  // Y
    x[2] = phi0;              // phi
    x[3] = vx0;               // vx
    // x[4..5] = vy, r = 0
    // x[6] = s = 0
    // x[7] = D  = 0 (no throttle)
    // x[8] = delta = 0
    x[9] = vx0;               // vs (initial progress speed = vx)

    // ── Simulation storage ────────────────────────────────────────────────
    // states logged as [X, Y, phi, s, vx] — compatible with existing plotter
    // ctrls logged as  [D, delta, vs]      — maps to plotter [v_cmd, theta, p]
    const int STEPS = 400;
    std::vector<std::vector<double>> states(STEPS + 1, std::vector<double>(5, 0.0));
    std::vector<std::vector<double>> ctrls(STEPS, std::vector<double>(3, 0.0));

    // Helper: map 10-D state → 5-D plotter format [X, Y, phi, s, vx]
    auto to_log_state = [](const std::vector<double>& x10) {
        return std::vector<double>{x10[0], x10[1], x10[2], x10[6], x10[3]};
    };

    states[0] = to_log_state(x);

    std::vector<casadi::DM>               pred_hist;
    std::vector<double>                   cost_hist, solve_time, t_cpu_s;
    std::vector<double>                   v_ref_hist;
    std::vector<double>                   cte_hist;
    std::vector<std::vector<double>>      u_pred_hist;
    std::vector<std::vector<ObstacleSpec>> obs_log;

    // Initial v_ref at s=0
    {
        double vr = td.v_ref_lut.is_null() ? param_num["v_max"] :
            static_cast<double>(td.v_ref_lut(std::vector<casadi::DM>{casadi::DM(0.0)})[0]);
        v_ref_hist.push_back(vr);
    }

    // ── Moving obstacles ──────────────────────────────────────────────────
    std::vector<MovingObstacle> moving_obs;
    if (USE_OBSTACLES) {
        // (s0, vs, radius, lat_offset)
        moving_obs.emplace_back(0.20 * td.s_total, 0.95, 0.25, 0.40);
        moving_obs.emplace_back(0.55 * td.s_total, 0.80, 0.20, -0.35);
    }

    // ── Closed-loop simulation ────────────────────────────────────────────
    for (int k = 0; k < STEPS; ++k) {

        // Step obstacles forward
        std::vector<MPC::Obstacle>  obstacles_solver;
        std::vector<ObstacleSpec>   obstacles_log_step;
        if (USE_OBSTACLES) {
            for (auto& mo : moving_obs) {
                mo.step(param_num["dT"], td.s_total);
                auto od = mo.as_dict(td);
                obstacles_solver.push_back({od.x, od.y, od.radius});
                obstacles_log_step.push_back(od);
            }
        }
        obs_log.push_back(obstacles_log_step);
        mpc.set_obstacles(obstacles_solver);

        // Solve
        auto sol = mpc.solve(x);
        pred_hist.push_back(sol.X_opt);
        cost_hist.push_back(mpc.last_cost);
        solve_time.push_back(mpc.last_solve_time);
        t_cpu_s.push_back(mpc.last_solve_time);

        // First optimal rates [dD, dDelta, dVs]
        const double dD_opt     = sol.u_cmd_first[0];
        const double dDelta_opt = sol.u_cmd_first[1];
        const double dVs_opt    = sol.u_cmd_first[2];

        // Collect predicted vs for CSV
        std::vector<double> p_pred;
        for (int i = 0; i < sol.U_opt.size1(); ++i)
            p_pred.push_back(static_cast<double>(sol.U_opt(i, 2)));
        u_pred_hist.push_back(p_pred);

        // Propagate one step with the dynamic model
        x = mpc.propagate(x, {dD_opt, dDelta_opt, dVs_opt});

        // Clamp state to physical bounds (guards against numerical blow-up)
        x[3] = std::clamp(x[3], 0.0,    param_num["v_max"]);  // vx
        x[4] = std::clamp(x[4], -3.0,   3.0);                  // vy
        x[5] = std::clamp(x[5], -8.0,   8.0);                  // r
        x[7] = std::clamp(x[7], -0.1,   1.0);                  // D
        x[8] = std::clamp(x[8], -param_num["theta_max"], param_num["theta_max"]); // delta
        x[9] = std::clamp(x[9], 0.0,    param_num["p_max"]);   // vs

        // Re-estimate s from actual (X,Y) to prevent integration drift
        auto s_pair = find_current_arc_length({x[0], x[1]}, P, S);
        x[6] = s_pair.first;

        // Log
        states[k+1] = to_log_state(x);
        // ctrls: actuator states after propagation [D, delta, vs]
        ctrls[k] = {x[7], x[8], x[9]};

        double s_mod = std::fmod(std::max(0.0, x[6]), td.s_total);
        double vr = td.v_ref_lut.is_null() ? param_num["v_max"] :
            static_cast<double>(td.v_ref_lut(std::vector<casadi::DM>{casadi::DM(s_mod)})[0]);
        v_ref_hist.push_back(vr);

        // Cross-track error for logging
        double dx_ref = static_cast<double>(td.c_lut_dx(std::vector<casadi::DM>{casadi::DM(s_mod)})[0]);
        double dy_ref = static_cast<double>(td.c_lut_dy(std::vector<casadi::DM>{casadi::DM(s_mod)})[0]);
        double x_ref  = static_cast<double>(td.c_lut_x(std::vector<casadi::DM>{casadi::DM(s_mod)})[0]);
        double y_ref  = static_cast<double>(td.c_lut_y(std::vector<casadi::DM>{casadi::DM(s_mod)})[0]);
        double cte = -dy_ref * (x[0] - x_ref) + dx_ref * (x[1] - y_ref);
        cte_hist.push_back(cte);

        std::cout << "Step " << k+1
                  << ": pos=(" << x[0] << "," << x[1] << ")"
                  << " vx=" << x[3] << " vs=" << x[9]
                  << " D="  << x[7] << " delta=" << x[8]
                  << " s="  << x[6] << "\n";
    }

    // ── Timing summary ────────────────────────────────────────────────────
    double t_sum = 0, t_min = 1e9, t_max = 0;
    for (double t : t_cpu_s) {
        t_sum += t;
        t_min = std::min(t_min, t);
        t_max = std::max(t_max, t);
    }
    double t_mean = t_sum / std::max<size_t>(1, t_cpu_s.size());
    std::cout << "--- min:  " << t_min  << " s\n";
    std::cout << "--- mean: " << t_mean << " s\n";
    std::cout << "--- max:  " << t_max  << " s\n";
    std::cout << "--- END ---\n";

    // ── CSV save ──────────────────────────────────────────────────────────
    auto save_info = save_run_csv_auto(
        "results", "mpcc_single_track",
        param_num["dT"],
        static_cast<int>(param_num["N"]),
        param_num["v_max"],
        states, ctrls,
        pred_hist, obs_log,
        cost_hist, solve_time, v_ref_hist, u_pred_hist,
        true);

    std::cout << "Saved run to: " << save_info.run_dir << "\n"
              << "  " << save_info.states_ctrls_csv << "\n"
              << "  " << save_info.predictions_csv  << "\n"
              << "  " << save_info.obstacles_csv     << "\n"
              << "  " << save_info.u_pred_csv        << "\n";

    // ── Plots (optional — requires working Python/matplotlib) ─────────────
    try {
        if (!fs::exists("plots")) fs::create_directory("plots");
        plot_outputs(states, v_ref_hist, "plots/mpcc_outputs.png");
        plot_inputs(ctrls, states,
                    -mpc.theta_max, mpc.theta_max,
                    param_num["p_min"], param_num["p_max"],
                    "plots/mpcc_inputs.png");
        plot_corridor(states, td.left, td.right, obs_log, "plots/mpcc_corridor.png");
        plot_timing(t_cpu_s, param_num["dT"], "plots/mpcc_timing.png");
        plot_progress(states, ctrls, "plots/mpcc_progress.png");
        plot_cost(cost_hist,          "plots/mpcc_cost.png");
        plot_tracking_errors(cte_hist, "plots/mpcc_cte.png");
        std::cout << "Plots saved to plots/\n";
    } catch (const std::exception& e) {
        std::cerr << "[Plotter] " << e.what() << "\n";
    }

    return 0;
}
