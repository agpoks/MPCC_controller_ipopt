//
// Created by poxx on 3/12/26.
//

#ifndef RL_MANAGER_H
#define RL_MANAGER_H
#pragma once

#include <casadi/casadi.hpp>
#include <string>
#include <vector>
#include <map>

struct TrackData {
    std::vector<std::vector<double>> center;
    std::vector<std::vector<double>> right;
    std::vector<std::vector<double>> left;

    std::vector<std::vector<double>> center_ext;
    std::vector<std::vector<double>> right_ext;
    std::vector<std::vector<double>> left_ext;
    std::vector<std::vector<double>> deriv_ext;

    std::vector<double> s_orig;
    std::vector<double> s_ext;
    double s_total = 0.0;

    casadi::Function c_lut_x;
    casadi::Function c_lut_y;
    casadi::Function c_lut_dx;
    casadi::Function c_lut_dy;
    casadi::Function r_lut_x;
    casadi::Function r_lut_y;
    casadi::Function l_lut_x;
    casadi::Function l_lut_y;
    casadi::Function v_ref_lut;
};

class RLManager {
public:
    explicit RLManager(const std::string& track_folder = "");

    TrackData load();
    TrackData load_from_csv(const std::string& folder);

    void build_reference_speed(double v_min,
                               double v_max,
                               double percentile = 95.0,
                               const std::string& mode = "linear");

    const TrackData& td() const { return td_; }

private:
    static std::vector<std::vector<double>> read_csv_xy(const std::string& filename);
    static std::vector<double> arc_lengths(const std::vector<std::vector<double>>& pts);
    static std::pair<casadi::Function, casadi::Function>
    mk_bspline(const std::string& label_x,
               const std::string& label_y,
               const std::vector<std::vector<double>>& pts,
               const std::vector<double>& arc);

    static std::vector<std::vector<double>>
    compute_unit_derivatives_closed(const std::vector<std::vector<double>>& center);

    static double percentile_of_abs(const std::vector<double>& values, double p);

private:
    std::string track_folder_;
    TrackData td_;
};
#endif //RL_MANAGER_H
