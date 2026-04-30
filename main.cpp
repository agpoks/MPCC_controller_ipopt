#include "main.h"
#include "MPCC_controller.h"
#include "rl_manager.h"
#include "utils.h"
#include "plotter.h"
#include "yaml_loader.h"

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Scenario configuration — everything that differs between runs
// ─────────────────────────────────────────────────────────────────────────────
struct ScenarioConfig {
    std::string name;                       // used as results folder prefix
    std::vector<MovingObstacle> obstacles;  // (empty = no obstacles)
    int steps = 400;                        // simulation steps
};

// ─────────────────────────────────────────────────────────────────────────────
// run_scenario — build MPC, simulate, save CSVs + plots + GIF
// ─────────────────────────────────────────────────────────────────────────────
static void run_scenario(
    const ScenarioConfig&           scenario,
    const TrackData&                td,
    const std::vector<std::vector<double>>& P,   // centre-line points
    const std::vector<double>&      S,            // cumulative arc-lengths
    std::map<std::string, double>&  param_num,
    const fs::path&                 project_dir)
{
    std::cout << "\n══════════════════════════════════════════════════════════\n"
              << "  Scenario: " << scenario.name << "\n"
              << "══════════════════════════════════════════════════════════\n";

    // ── Re-build MPC for each scenario (parameters may differ) ──────────
    MPC mpc;
    mpc.set_initial_params(param_num, {});
    mpc.set_track_data(
        td.c_lut_x, td.c_lut_y, td.c_lut_dx, td.c_lut_dy,
        td.r_lut_x, td.r_lut_y, td.l_lut_x,  td.l_lut_y,
        td.s_ext, td.s_total, td.v_ref_lut);
    mpc.setup_MPC();

    // ── Initial vehicle state [X, Y, phi, vx, vy, r, s, D, delta, vs] ──
    const double vx0 = 0.3;
    const double phi0 = std::atan2(
        static_cast<double>(td.c_lut_dy(std::vector<casadi::DM>{casadi::DM(0.0)})[0]),
        static_cast<double>(td.c_lut_dx(std::vector<casadi::DM>{casadi::DM(0.0)})[0]));

    std::vector<double> x(NX, 0.0);
    x[0] = td.center[0][0];
    x[1] = td.center[0][1];
    x[2] = phi0;
    x[3] = vx0;
    x[9] = vx0;  // vs starts at initial speed

    // ── Simulation log storage ────────────────────────────────────────────
    // states: [X, Y, phi, s, vx]   (5-element, compatible with plotter)
    // ctrls : [D, delta, vs]        (actuator states after each propagation)
    const int STEPS = scenario.steps;
    std::vector<std::vector<double>> states(STEPS + 1, std::vector<double>(5, 0.0));
    std::vector<std::vector<double>> ctrls(STEPS,       std::vector<double>(3, 0.0));

    auto to_log_state = [](const std::vector<double>& x10) {
        return std::vector<double>{x10[0], x10[1], x10[2], x10[6], x10[3]};
    };
    states[0] = to_log_state(x);

    std::vector<casadi::DM>                pred_hist;
    std::vector<double>                    cost_hist, solve_time, t_cpu_s;
    std::vector<double>                    v_ref_hist, cte_hist;
    std::vector<std::vector<double>>       u_pred_hist;
    std::vector<std::vector<ObstacleSpec>> obs_log;

    // Initial v_ref at s = 0
    {
        double vr = td.v_ref_lut.is_null() ? param_num["v_max"]
                  : static_cast<double>(td.v_ref_lut(
                        std::vector<casadi::DM>{casadi::DM(0.0)})[0]);
        v_ref_hist.push_back(vr);
    }

    // Mutable obstacle list for this scenario (copy so we can step them)
    std::vector<MovingObstacle> moving_obs = scenario.obstacles;

    // ── Closed-loop simulation ─────────────────────────────────────────────
    for (int k = 0; k < STEPS; ++k) {

        // Advance obstacles and build solver / log lists
        std::vector<MPC::Obstacle>   obstacles_solver;
        std::vector<ObstacleSpec>    obstacles_log_step;
        for (auto& mo : moving_obs) {
            mo.step(param_num["dT"], td.s_total);
            auto od = mo.as_dict(td);
            obstacles_solver.push_back({od.x, od.y, od.radius});
            obstacles_log_step.push_back(od);
        }
        obs_log.push_back(obstacles_log_step);
        mpc.set_obstacles(obstacles_solver);

        // Solve one MPC step
        auto sol = mpc.solve(x);
        pred_hist.push_back(sol.X_opt);
        cost_hist.push_back(mpc.last_cost);
        solve_time.push_back(mpc.last_solve_time);
        t_cpu_s.push_back(mpc.last_solve_time);

        // Collect predicted virtual speed (vs) over the horizon for diagnostics
        std::vector<double> p_pred;
        for (int i = 0; i < sol.X_opt.size1(); ++i)
            p_pred.push_back(static_cast<double>(sol.X_opt(i, 9)));
        u_pred_hist.push_back(p_pred);

        // Apply first optimal input and integrate one step
        const double dD_opt     = sol.u_cmd_first[0];
        const double dDelta_opt = sol.u_cmd_first[1];
        const double dVs_opt    = sol.u_cmd_first[2];
        x = mpc.propagate(x, {dD_opt, dDelta_opt, dVs_opt});

        // Clamp state to physical bounds (prevents numerical blow-up)
        x[3] = std::clamp(x[3],  0.0,  param_num["v_max"]);
        x[4] = std::clamp(x[4], -5.0,  5.0);
        x[5] = std::clamp(x[5], -10.0, 10.0);
        {
            double d_min = param_num.count("D_min") ? param_num.at("D_min") : -0.1;
            double d_max = param_num.count("D_max") ? param_num.at("D_max") :  1.0;
            x[7] = std::clamp(x[7], d_min, d_max);
        }
        x[8] = std::clamp(x[8], -param_num["theta_max"], param_num["theta_max"]);
        x[9] = std::clamp(x[9],  0.0,  param_num["p_max"]);

        // Re-estimate arc-length from actual (X,Y) to prevent integration drift
        x[6] = find_current_arc_length({x[0], x[1]}, P, S).first;

        states[k+1] = to_log_state(x);
        ctrls[k]    = {x[7], x[8], x[9]};

        // Reference speed and CTE at updated position
        double s_mod = std::fmod(std::max(0.0, x[6]), td.s_total);
        double vr    = td.v_ref_lut.is_null() ? param_num["v_max"]
                     : static_cast<double>(td.v_ref_lut(
                           std::vector<casadi::DM>{casadi::DM(s_mod)})[0]);
        v_ref_hist.push_back(vr);

        double dx_ref = static_cast<double>(td.c_lut_dx(std::vector<casadi::DM>{casadi::DM(s_mod)})[0]);
        double dy_ref = static_cast<double>(td.c_lut_dy(std::vector<casadi::DM>{casadi::DM(s_mod)})[0]);
        double x_ref  = static_cast<double>(td.c_lut_x( std::vector<casadi::DM>{casadi::DM(s_mod)})[0]);
        double y_ref  = static_cast<double>(td.c_lut_y( std::vector<casadi::DM>{casadi::DM(s_mod)})[0]);
        cte_hist.push_back(-dy_ref*(x[0]-x_ref) + dx_ref*(x[1]-y_ref));

        if ((k+1) % 50 == 0) {
            std::cout << "  step " << k+1 << "/" << STEPS
                      << "  pos=(" << x[0] << "," << x[1] << ")"
                      << "  vx=" << x[3] << "  s=" << x[6]
                      << "  solve=" << mpc.last_solve_time*1e3 << "ms\n";
        }
    }

    // ── Timing summary ────────────────────────────────────────────────────
    double t_sum=0, t_min=1e9, t_max=0;
    for (double t : t_cpu_s) {
        t_sum += t;
        t_min = std::min(t_min, t);
        t_max = std::max(t_max, t);
    }
    double t_mean = t_sum / std::max<size_t>(1, t_cpu_s.size());
    std::cout << "  Solve times  min=" << t_min*1e3  << "ms"
              << "  mean=" << t_mean*1e3 << "ms"
              << "  max=" << t_max*1e3  << "ms\n";
    if (t_mean > 1.0/50.0)
        std::cout << "  WARNING: mean solve time > 20ms — not real-time at 50Hz.\n"
                  << "           Reduce N or tighten ipopt_acceptable_tol.\n";

    // ── Save CSVs ──────────────────────────────────────────────────────────
    LoggedLimits lim;
    lim.theta_min = -param_num["theta_max"];
    lim.theta_max =  param_num["theta_max"];
    lim.vx_min    = 0.0;
    lim.vx_max    = param_num["v_max"];
    lim.D_min     = param_num.count("D_min") ? param_num.at("D_min") : -0.1;
    lim.D_max     = param_num.count("D_max") ? param_num.at("D_max") :  1.0;
    lim.vs_min    = param_num["p_min"];
    lim.vs_max    = param_num["p_max"];

    auto save_info = save_run_csv_auto(
        "results", scenario.name,
        param_num["dT"],
        static_cast<int>(param_num["N"]),
        param_num["v_max"],
        states, ctrls,
        pred_hist, obs_log,
        cost_hist, solve_time, v_ref_hist, u_pred_hist,
        lim, /*timestamp=*/true);

    std::cout << "  Results → " << save_info.run_dir << "\n";

    // ── Static plots ───────────────────────────────────────────────────────
    const std::string plot_dir = save_info.run_dir + "/plots";
    try {
        if (!fs::exists(plot_dir)) fs::create_directories(plot_dir);
        plot_outputs(states, v_ref_hist,                 plot_dir + "/outputs.png");
        plot_inputs(ctrls, states,
                    lim.theta_min, lim.theta_max,
                    lim.vs_min,    lim.vs_max,           plot_dir + "/inputs.png");
        plot_corridor(states, td.left, td.right, obs_log, plot_dir + "/corridor.png");
        plot_timing(t_cpu_s, param_num["dT"],             plot_dir + "/timing.png");
        plot_progress(states, ctrls,                      plot_dir + "/progress.png");
        plot_cost(cost_hist,                              plot_dir + "/cost.png");
        plot_tracking_errors(cte_hist,                    plot_dir + "/cte.png");
        std::cout << "  Plots → " << plot_dir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "  [Plotter] " << e.what() << "\n";
    }

    // ── Playback GIF ──────────────────────────────────────────────────────
    try {
        fs::path script = project_dir / "scripts" / "playback_dashboard_csv.py";
        fs::path gif_out = fs::path(save_info.run_dir) / "playback.gif";
        std::string cmd = "python3 \"" + script.string() + "\""
                        + " --results-dir \"" + save_info.run_dir + "\""
                        + " --track-folder \"" + (project_dir / "raceline").string() + "\""
                        + " --gif \"" + gif_out.string() + "\""
                        + " --fps 15 --dpi 100 --gif-max-frames 300"
                        + " --no-show";
        std::cout << "  Generating GIF (this may take ~30s)…\n";
        int ret = std::system(cmd.c_str());
        if (ret == 0) std::cout << "  GIF → " << gif_out.string() << "\n";
        else          std::cerr << "  GIF generation failed (exit " << ret << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "  [GIF] " << e.what() << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    const fs::path project_dir = fs::path(__FILE__).parent_path();

    // ── Load YAML configuration ───────────────────────────────────────────
    std::map<std::string, double>      param_num;
    std::map<std::string, std::string> param_str;

    auto load_yaml_to_num = [&](const std::string& path) {
        try {
            auto flat = load_yaml_flat(path);
            for (auto& [k, v] : flat) {
                try { param_num[k] = std::stod(v); } catch (...) {}
            }
            std::cout << "[main] Loaded " << path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[main] YAML warning: " << e.what() << " — using defaults\n";
        }
    };

    // Vehicle parameters use "vp_" prefix so MPC::set_initial_params can read them.
    // Rename Cr_tire → vp_Cr_tire after loading to match MPC expectations.
    {
        auto veh = load_yaml_flat((project_dir / "params/vehicle.yaml").string());
        for (auto& [k, v] : veh) {
            // Rename Cr_tire to vp_Cr_tire; prefix all others with vp_
            std::string key = (k == "Cr_tire") ? "vp_Cr_tire" : "vp_" + k;
            try { param_num[key] = std::stod(v); } catch (...) {}
        }
        std::cout << "[main] Loaded params/vehicle.yaml\n";
    }
    load_yaml_to_num((project_dir / "params/mpcc_tuning.yaml").string());

    // ── Track loading ─────────────────────────────────────────────────────
    const fs::path track_folder = project_dir / "raceline";
    if (!fs::exists(track_folder))
        throw std::runtime_error("Track folder not found: " + track_folder.string());

    RLManager rl(track_folder.string());
    TrackData td = rl.load();
    {
        double vref_min = param_num.count("v_ref_min") ? param_num.at("v_ref_min") : 1.5;
        rl.build_reference_speed(vref_min, param_num["v_max"], 95.0, "linear");
    }
    td = rl.td();
    param_num["s_max"] = td.s_ext.back();

    // Centre-line waypoints for s re-estimation after each simulation step
    auto& P = td.center;
    std::vector<double> S(P.size(), 0.0);
    for (size_t i = 1; i < P.size(); ++i) {
        double dx = P[i][0]-P[i-1][0], dy = P[i][1]-P[i-1][1];
        S[i] = S[i-1] + std::sqrt(dx*dx + dy*dy);
    }
    if (!S.empty()) S.back() = td.s_total;

    std::cout << "[main] Track loaded: length=" << td.s_total << " m\n";

    // ── Scenario 1 — standard racing with two small moving obstacles ──────
    //   Two obstacles at ~20% and ~55% of the track, moving faster than ~0.5×
    //   the target speed; small radii allow the car to pass on either side.
    {
        ScenarioConfig sc;
        sc.name  = "scenario_standard";
        sc.steps = 400;
        sc.obstacles.emplace_back(0.20 * td.s_total, 0.95, 0.25,  0.40);  // right-ish
        sc.obstacles.emplace_back(0.55 * td.s_total, 0.80, 0.20, -0.35);  // left-ish
        run_scenario(sc, td, P, S, param_num, project_dir);
    }

    // ── Scenario 2 — large slow obstacle forcing deceleration + overtake ─
    //   One large obstacle (r=0.45 m) starts at 25% of the track and moves at
    //   0.35 m/s (≈ 10% of v_max).  At track widths < ~2 m the car cannot pass;
    //   it decelerates and follows until the track widens, then overtakes.
    //   Runs for 600 steps (~30 s) to capture the full overtaking manoeuvre.
    {
        ScenarioConfig sc;
        sc.name  = "scenario_large_obs";
        sc.steps = 600;
        sc.obstacles.emplace_back(0.25 * td.s_total, 0.35, 0.45, 0.05);
        run_scenario(sc, td, P, S, param_num, project_dir);
    }

    // ── Car7 scenarios — run if car7 parameter files are present ─────────────
    {
        const fs::path car7_veh  = project_dir / "params/car7_vehicle.yaml";
        const fs::path car7_tune = project_dir / "params/car7_mpcc_tuning.yaml";

        if (fs::exists(car7_veh) && fs::exists(car7_tune)) {
            std::cout << "\n[main] Car7 params found — running car7 scenarios\n";

            std::map<std::string, double> param7;

            // Vehicle params with vp_ prefix (same rename logic as standard vehicle.yaml)
            {
                auto veh = load_yaml_flat(car7_veh.string());
                for (auto& [k, v] : veh) {
                    std::string key = (k == "Cr_tire") ? "vp_Cr_tire" : "vp_" + k;
                    try { param7[key] = std::stod(v); } catch (...) {}
                }
                std::cout << "[main] Loaded " << car7_veh.string() << "\n";
            }

            // Tuning params
            {
                auto flat = load_yaml_flat(car7_tune.string());
                for (auto& [k, v] : flat) {
                    try { param7[k] = std::stod(v); } catch (...) {}
                }
                std::cout << "[main] Loaded " << car7_tune.string() << "\n";
            }

            param7["s_max"] = td.s_ext.back();

            // Rebuild curvature-based reference speed for car7 v_max
            double vref_min7 = param7.count("v_ref_min") ? param7.at("v_ref_min") : 2.0;
            rl.build_reference_speed(vref_min7, param7.at("v_max"), 95.0, "linear");
            TrackData td7 = rl.td();

            // ── Car7 Scenario 1 — standard: 2 small moving obstacles ────────
            {
                ScenarioConfig sc;
                sc.name  = "car7_scenario_standard";
                sc.steps = 600;
                sc.obstacles.emplace_back(0.20 * td7.s_total, 0.70, 0.25,  0.40);
                sc.obstacles.emplace_back(0.55 * td7.s_total, 0.80, 0.20, -0.35);
                run_scenario(sc, td7, P, S, param7, project_dir);
            }

            // ── Car7 Scenario 2 — large slow obstacle: follow then overtake ─
            {
                ScenarioConfig sc;
                sc.name  = "car7_scenario_large_obs";
                sc.steps = 600;
                sc.obstacles.emplace_back(0.25 * td7.s_total, 0.50, 0.45, 0.05);
                run_scenario(sc, td7, P, S, param7, project_dir);
            }

            // ── Car7 Scenario 3 — triple obstacles, one nearly car-speed ────
            //   obs1: large slow (r=0.45, vs=0.5) — blocking
            //   obs2: medium speed (r=0.38, vs=2.5) — moderate challenge
            //   obs3: fast (r=0.33, vs=5.5 ≈ 79% of v_max) — nearly as fast
            {
                ScenarioConfig sc;
                sc.name  = "car7_scenario_triple_obs";
                sc.steps = 600;
                sc.obstacles.emplace_back(0.15 * td7.s_total, 0.50, 0.45,  0.00);
                sc.obstacles.emplace_back(0.40 * td7.s_total, 2.50, 0.38, -0.15);
                sc.obstacles.emplace_back(0.65 * td7.s_total, 5.50, 0.33,  0.05);
                run_scenario(sc, td7, P, S, param7, project_dir);
            }
        }
    }

    std::cout << "\n[main] All scenarios done.\n";
    return 0;
}
