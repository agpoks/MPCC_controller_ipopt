//
// Created by poxx on 3/12/26.
//
#include "rl_manager.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <filesystem>

using namespace casadi;
namespace fs = std::filesystem;

RLManager::RLManager(const std::string& track_folder)
    : track_folder_(track_folder) {}

std::vector<std::vector<double>> RLManager::read_csv_xy(const std::string& filename) {
    if (!fs::exists(filename)) {
        throw std::runtime_error("File not found: " + filename);
    }

    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<double>> data;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string item;
        std::vector<double> row;

        while (std::getline(ss, item, ',')) {
            row.push_back(std::stod(item));
        }
        if (row.size() >= 2) {
            data.push_back({row[0], row[1]});
        }
    }

    if (data.empty()) {
        throw std::runtime_error("CSV is empty: " + filename);
    }
    return data;
}

std::vector<double> RLManager::arc_lengths(const std::vector<std::vector<double>>& pts) {
    std::vector<double> s;
    s.reserve(pts.size());
    s.push_back(0.0);

    for (size_t i = 1; i < pts.size(); ++i) {
        double dx = pts[i][0] - pts[i - 1][0];
        double dy = pts[i][1] - pts[i - 1][1];
        double seg = std::sqrt(dx * dx + dy * dy);
        s.push_back(s.back() + seg);
    }
    return s;
}

std::pair<Function, Function>
RLManager::mk_bspline(const std::string& label_x,
                      const std::string& label_y,
                      const std::vector<std::vector<double>>& pts,
                      const std::vector<double>& arc) {
    std::vector<double> vx(pts.size()), vy(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        vx[i] = pts[i][0];
        vy[i] = pts[i][1];
    }

    std::vector<std::vector<double>> grid = {arc};
    Dict opts;

    Function lut_x = interpolant(label_x, "bspline", grid, vx, opts);
    Function lut_y = interpolant(label_y, "bspline", grid, vy, opts);
    return {lut_x, lut_y};
}

std::vector<std::vector<double>>
RLManager::compute_unit_derivatives_closed(const std::vector<std::vector<double>>& center) {
    std::vector<std::vector<double>> deriv(center.size(), std::vector<double>(2, 0.0));

    for (size_t i = 0; i < center.size(); ++i) {
        size_t j = (i + 1) % center.size();
        double dx = center[j][0] - center[i][0];
        double dy = center[j][1] - center[i][1];
        double nrm = std::sqrt(dx * dx + dy * dy) + 1e-9;
        deriv[i][0] = dx / nrm;
        deriv[i][1] = dy / nrm;
    }
    return deriv;
}

TrackData RLManager::load_from_csv(const std::string& folder) {
    TrackData td;

    td.center = read_csv_xy(folder + "/centerline_waypoints.csv");
    td.right  = read_csv_xy(folder + "/right_waypoints.csv");
    td.left   = read_csv_xy(folder + "/left_waypoints.csv");

    std::string deriv_file = folder + "/center_spline_derivatives.csv";
    std::vector<std::vector<double>> deriv;
    if (fs::exists(deriv_file)) {
        deriv = read_csv_xy(deriv_file);
    } else {
        deriv = compute_unit_derivatives_closed(td.center);
    }

    size_t ext_start = 1;
    size_t ext_end = 1 + td.center.size() / 2;

    td.center_ext = td.center;
    td.right_ext  = td.right;
    td.left_ext   = td.left;
    td.deriv_ext  = deriv;

    for (size_t i = ext_start; i < ext_end && i < td.center.size(); ++i) {
        td.center_ext.push_back(td.center[i]);
        td.right_ext.push_back(td.right[i]);
        td.left_ext.push_back(td.left[i]);
        td.deriv_ext.push_back(deriv[i]);
    }

    td.s_orig = arc_lengths(td.center);
    td.s_ext  = arc_lengths(td.center_ext);
    td.s_total = td.s_orig.back();

    auto cxy = mk_bspline("c_x", "c_y", td.center_ext, td.s_ext);
    auto cdxdy = mk_bspline("c_dx", "c_dy", td.deriv_ext, td.s_ext);
    auto rxy = mk_bspline("r_x", "r_y", td.right_ext, td.s_ext);
    auto lxy = mk_bspline("l_x", "l_y", td.left_ext, td.s_ext);

    td.c_lut_x = cxy.first;
    td.c_lut_y = cxy.second;
    td.c_lut_dx = cdxdy.first;
    td.c_lut_dy = cdxdy.second;
    td.r_lut_x = rxy.first;
    td.r_lut_y = rxy.second;
    td.l_lut_x = lxy.first;
    td.l_lut_y = lxy.second;

    td_ = td;
    return td_;
}

TrackData RLManager::load() {
    if (track_folder_.empty()) {
        throw std::runtime_error("track_folder is empty");
    }
    return load_from_csv(track_folder_);
}

double RLManager::percentile_of_abs(const std::vector<double>& values, double p) {
    if (values.empty()) return 1.0;
    std::vector<double> tmp = values;
    std::sort(tmp.begin(), tmp.end());
    double pos = (p / 100.0) * (static_cast<double>(tmp.size() - 1));
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return tmp[lo];
    double alpha = pos - static_cast<double>(lo);
    return (1.0 - alpha) * tmp[lo] + alpha * tmp[hi];
}

void RLManager::build_reference_speed(double v_min,
                                      double v_max,
                                      double percentile,
                                      const std::string& mode) {
    if (td_.s_ext.empty()) {
        throw std::runtime_error("Track must be loaded before build_reference_speed()");
    }

    std::vector<double> kappa(td_.s_ext.size(), 0.0);

    for (size_t i = 0; i < td_.s_ext.size(); ++i) {
        double s = td_.s_ext[i];
        double tx = static_cast<double>(td_.c_lut_dx(std::vector<DM>{DM(s)})[0]);
        double ty = static_cast<double>(td_.c_lut_dy(std::vector<DM>{DM(s)})[0]);

        size_t j = (i + 1 < td_.s_ext.size()) ? i + 1 : i;
        double ds = std::max(1e-6, td_.s_ext[j] - td_.s_ext[i]);
        double s2 = std::min(td_.s_ext.back(), s + ds);

        double tx2 = static_cast<double>(td_.c_lut_dx(std::vector<DM>{DM(s2)})[0]);
        double ty2 = static_cast<double>(td_.c_lut_dy(std::vector<DM>{DM(s2)})[0]);

        double dtx = (tx2 - tx) / ds;
        double dty = (ty2 - ty) / ds;
        double denom = std::pow(tx * tx + ty * ty, 1.5) + 1e-9;
        kappa[i] = std::abs(tx * dty - ty * dtx) / denom;
    }

    double k_ref = std::max(1e-9, percentile_of_abs(kappa, percentile));
    std::vector<double> vref(td_.s_ext.size(), v_max);

    for (size_t i = 0; i < vref.size(); ++i) {
        if (mode == "linear") {
            double alpha = std::min(1.0, kappa[i] / k_ref);
            vref[i] = v_max - alpha * (v_max - v_min);
        } else {
            vref[i] = v_max;
        }
        vref[i] = std::max(v_min, std::min(v_max, vref[i]));
    }

    std::vector<std::vector<double>> grid = {td_.s_ext};
    Dict opts;
    td_.v_ref_lut = interpolant("v_ref_lut", "bspline", grid, vref, opts);
}