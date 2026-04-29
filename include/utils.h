#pragma once
#include <vector>
#include <string>
#include <utility>
#include <casadi/casadi.hpp>

// Forward declaration — full definition in rl_manager.h
struct TrackData;

struct ObstacleSpec {
    double x, y, radius;
};

struct SaveRunResult {
    std::string run_dir;
    std::string states_ctrls_csv;
    std::string predictions_csv;
    std::string obstacles_csv;
    std::string u_pred_csv;
};

struct LoggedLimits {
    double theta_min = -0.35;
    double theta_max = 0.35;
    double vx_min = 0.0;
    double vx_max = 3.5;
    double D_min = -0.1;
    double D_max = 1.0;
    double vs_min = 0.0;
    double vs_max = 3.5;
};

// Obstacle that moves along the track centerline at constant speed
class MovingObstacle {
public:
    MovingObstacle(double s0, double vs, double radius, double lat_offset)
        : s(s0), vs(vs), radius(radius), lat_offset(lat_offset) {}

    void step(double dt, double s_total);
    std::pair<double, double> pose_xy(const TrackData& td) const;
    ObstacleSpec as_dict(const TrackData& td) const;

    double s;           // arc-length position along track
    double vs;          // speed along track [m/s]
    double radius;      // obstacle radius [m]
    double lat_offset;  // lateral offset from centerline [m], + = left
};

std::pair<int, double> find_nearest_index(const std::vector<std::vector<double>>& P,
                                          const std::vector<double>& car_pos);

std::pair<double, int> find_current_arc_length(const std::vector<double>& car_pos,
                                               const std::vector<std::vector<double>>& P,
                                               const std::vector<double>& S);

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
    bool timestamp = true
);
