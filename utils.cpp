//
// Created by poxx on 5/15/25.
//
#include "utils.h"
#include "rl_manager.h"   // full TrackData definition for MovingObstacle::pose_xy

#include <casadi/casadi.hpp>

#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>

namespace fs = std::filesystem;

using namespace casadi;

void MovingObstacle::step(double dt, double s_total) {
    s += vs * dt;
    if (s >= s_total) {
        s -= s_total;
    } else if (s < 0.0) {
        s += s_total;
    }
}

std::pair<double, double> MovingObstacle::pose_xy(const TrackData& td) const {
    double cx = static_cast<double>(td.c_lut_x(std::vector<DM>{DM(s)})[0]);
    double cy = static_cast<double>(td.c_lut_y(std::vector<DM>{DM(s)})[0]);
    double dx = static_cast<double>(td.c_lut_dx(std::vector<DM>{DM(s)})[0]);
    double dy = static_cast<double>(td.c_lut_dy(std::vector<DM>{DM(s)})[0]);

    double nrm = std::max(1e-9, std::sqrt(dx * dx + dy * dy));
    double nx = -dy / nrm;
    double ny =  dx / nrm;

    return {cx + lat_offset * nx, cy + lat_offset * ny};
}

ObstacleSpec MovingObstacle::as_dict(const TrackData& td) const {
    auto xy = pose_xy(td);
    return {xy.first, xy.second, radius};
}

std::pair<int, double> find_nearest_index(const std::vector<std::vector<double>>& P,
                                          const std::vector<double>& car_pos) {
    int idx = 0;
    double best_d2 = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < P.size(); ++i) {
        double dx = P[i][0] - car_pos[0];
        double dy = P[i][1] - car_pos[1];
        double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            idx = static_cast<int>(i);
        }
    }
    return {idx, std::sqrt(best_d2)};
}

std::pair<double, int> find_current_arc_length(const std::vector<double>& car_pos,
                                               const std::vector<std::vector<double>>& P,
                                               const std::vector<double>& S) {
    auto nearest = find_nearest_index(P, car_pos);
    int nearest_index = nearest.first;
    double minimum_dist = nearest.second;
    int N = static_cast<int>(P.size());

    const double ARC_LENGTH_MIN_DIST_TOL = 0.05;
    double current_s = 0.0;

    if (minimum_dist > ARC_LENGTH_MIN_DIST_TOL) {
        int prev_idx, next_idx;
        if (nearest_index == 0) {
            prev_idx = N - 1; next_idx = 1;
        } else if (nearest_index == N - 1) {
            prev_idx = N - 2; next_idx = 0;
        } else {
            prev_idx = nearest_index - 1;
            next_idx = nearest_index + 1;
        }

        double v_prev_x = P[prev_idx][0] - P[nearest_index][0];
        double v_prev_y = P[prev_idx][1] - P[nearest_index][1];

        double dp =
            (car_pos[0] - P[nearest_index][0]) * v_prev_x +
            (car_pos[1] - P[nearest_index][1]) * v_prev_y;

        int i_start, i_end;
        if (dp > 0.0) {
            i_start = prev_idx;
            i_end = nearest_index;
        } else {
            i_start = nearest_index;
            i_end = next_idx;
        }

        double seg_x = P[i_end][0] - P[i_start][0];
        double seg_y = P[i_end][1] - P[i_start][1];
        double seg_len = std::sqrt(seg_x * seg_x + seg_y * seg_y) + 1e-12;

        double new_dot =
            (car_pos[0] - P[i_start][0]) * seg_x +
            (car_pos[1] - P[i_start][1]) * seg_y;

        double projection = new_dot / seg_len;
        current_s = S[i_start] + projection;
    } else {
        current_s = S[nearest_index];
    }

    if (nearest_index == 0) {
        current_s = 0.0;
    }

    return {current_s, nearest_index};
}


static std::string make_timestamp_string()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t_c);
#else
    localtime_r(&t_c, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d_%H%M%S");
    return oss.str();
}

static void ensure_dir_exists(const fs::path& dir)
{
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
}

static double safe_at(const std::vector<double>& v, std::size_t i, double fallback = std::numeric_limits<double>::quiet_NaN())
{
    return (i < v.size()) ? v[i] : fallback;
}

SaveRunResult save_run_csv_auto(
    const std::string& base_results_dir,
    const std::string& run_name,
    double dT,
    int N,
    double ref_vel,
    const std::vector<std::vector<double>>& states,
    const std::vector<std::vector<double>>& ctrls,
    const std::vector<casadi::DM>& pred_hist,
    const std::vector<std::vector<ObstacleSpec>>& obs_log,
    const std::vector<double>& cost_hist,
    const std::vector<double>& solve_time,
    const std::vector<double>& v_ref_series,
    const std::vector<std::vector<double>>& u_pred_hist,
    const LoggedLimits& limits,
    bool timestamp
)
{
    if (states.empty()) {
        throw std::runtime_error("save_run_csv_auto: states is empty");
    }
    if (ctrls.empty()) {
        throw std::runtime_error("save_run_csv_auto: ctrls is empty");
    }

    fs::path base_dir(base_results_dir);
    ensure_dir_exists(base_dir);

    fs::path run_dir;
    if (timestamp) {
        run_dir = base_dir / (run_name + "_" + make_timestamp_string());
    } else {
        run_dir = base_dir / run_name;
    }
    ensure_dir_exists(run_dir);

    fs::path states_ctrls_csv = run_dir / "states_ctrls.csv";
    fs::path predictions_csv  = run_dir / "predictions.csv";
    fs::path obstacles_csv    = run_dir / "obstacles.csv";
    fs::path u_pred_csv       = run_dir / "u_pred.csv";

    // ------------------------------------------------------------
    // 1) states_ctrls.csv
    // ------------------------------------------------------------
    {
        std::ofstream f(states_ctrls_csv);
        if (!f) {
            throw std::runtime_error("Could not open " + states_ctrls_csv.string());
        }

                f << "k,t,x,y,psi,s,v_state,v_cmd,theta,p_cmd,cost,solve_time,v_ref,"
                    << "theta_min,theta_max,vx_min,vx_max,D_min,D_max,vs_min,vs_max\n";

        const std::size_t steps = ctrls.size();
        for (std::size_t k = 0; k < steps; ++k) {
            double t = static_cast<double>(k) * dT;

            double x = (k < states.size() && states[k].size() > 0) ? states[k][0] : std::numeric_limits<double>::quiet_NaN();
            double y = (k < states.size() && states[k].size() > 1) ? states[k][1] : std::numeric_limits<double>::quiet_NaN();
            double psi = (k < states.size() && states[k].size() > 2) ? states[k][2] : std::numeric_limits<double>::quiet_NaN();
            double s = (k < states.size() && states[k].size() > 3) ? states[k][3] : std::numeric_limits<double>::quiet_NaN();
            double v_state = (k < states.size() && states[k].size() > 4) ? states[k][4] : std::numeric_limits<double>::quiet_NaN();

            double v_cmd = (ctrls[k].size() > 0) ? ctrls[k][0] : std::numeric_limits<double>::quiet_NaN();
            double theta = (ctrls[k].size() > 1) ? ctrls[k][1] : std::numeric_limits<double>::quiet_NaN();
            double p_cmd = (ctrls[k].size() > 2) ? ctrls[k][2] : std::numeric_limits<double>::quiet_NaN();

            double cost = safe_at(cost_hist, k);
            double stime = safe_at(solve_time, k);
            double vref = safe_at(v_ref_series, k, ref_vel);

            f << k << ","
              << t << ","
              << x << ","
              << y << ","
              << psi << ","
              << s << ","
              << v_state << ","
              << v_cmd << ","
              << theta << ","
              << p_cmd << ","
              << cost << ","
              << stime << ","
              << vref << ","
              << limits.theta_min << ","
              << limits.theta_max << ","
              << limits.vx_min << ","
              << limits.vx_max << ","
              << limits.D_min << ","
              << limits.D_max << ","
              << limits.vs_min << ","
              << limits.vs_max << "\n";
        }
    }

    // ------------------------------------------------------------
    // 2) predictions.csv
    // ------------------------------------------------------------
    {
        std::ofstream f(predictions_csv);
        if (!f) {
            throw std::runtime_error("Could not open " + predictions_csv.string());
        }

        f << "step,pred_idx,x,y,psi,s\n";

        for (std::size_t step = 0; step < pred_hist.size(); ++step) {
            const casadi::DM& pred = pred_hist[step];
            if (pred.is_empty()) continue;

            const int rows = static_cast<int>(pred.size1());
            const int cols = static_cast<int>(pred.size2());
            // Need at least 7 columns: state vector [X,Y,phi,vx,vy,r,s,...]
            // Column 6 is the arc-length s; column 3 is vx (not s).
            if (cols < 7) continue;

            for (int i = 0; i < rows; ++i) {
                f << step << ","
                  << i << ","
                  << static_cast<double>(pred(i, 0)) << ","   // X
                  << static_cast<double>(pred(i, 1)) << ","   // Y
                  << static_cast<double>(pred(i, 2)) << ","   // phi (heading)
                  << static_cast<double>(pred(i, 6)) << "\n"; // s  (arc-length)
            }
        }
    }

    // ------------------------------------------------------------
    // 3) obstacles.csv
    // ------------------------------------------------------------
    {
        std::ofstream f(obstacles_csv);
        if (!f) {
            throw std::runtime_error("Could not open " + obstacles_csv.string());
        }

        f << "step,obs_id,x,y,radius\n";

        for (std::size_t step = 0; step < obs_log.size(); ++step) {
            for (std::size_t obs_id = 0; obs_id < obs_log[step].size(); ++obs_id) {
                const auto& obs = obs_log[step][obs_id];
                f << step << ","
                  << obs_id << ","
                  << obs.x << ","
                  << obs.y << ","
                  << obs.radius << "\n";
            }
        }
    }

    // ------------------------------------------------------------
    // 4) u_pred.csv
    // ------------------------------------------------------------
    {
        std::ofstream f(u_pred_csv);
        if (!f) {
            throw std::runtime_error("Could not open " + u_pred_csv.string());
        }

        f << "step,pred_idx,v_pred\n";

        for (std::size_t step = 0; step < u_pred_hist.size(); ++step) {
            const auto& up = u_pred_hist[step];
            for (std::size_t i = 0; i < up.size(); ++i) {
                f << step << ","
                  << i << ","
                  << up[i] << "\n";
            }
        }
    }

    SaveRunResult out;
    out.run_dir = run_dir.string();
    out.states_ctrls_csv = states_ctrls_csv.string();
    out.predictions_csv = predictions_csv.string();
    out.obstacles_csv = obstacles_csv.string();
    out.u_pred_csv = u_pred_csv.string();
    return out;
}