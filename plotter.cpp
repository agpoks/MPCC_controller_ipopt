#include "plotter.h"
#include "matplotlibcpp.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace plt = matplotlibcpp;

static std::vector<double> make_index_vector(std::size_t n) {
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i);
    }
    return x;
}

static std::vector<double> get_col(const std::vector<std::vector<double>>& data, std::size_t col) {
    std::vector<double> out(data.size(), 0.0);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i].size() > col) out[i] = data[i][col];
    }
    return out;
}

static std::vector<double> constant_vec(std::size_t n, double val) {
    return std::vector<double>(n, val);
}

void plot_outputs(const std::vector<std::vector<double>>& states,
                  const std::vector<double>& v_ref,
                  const std::string& filename)
{
    if (states.empty()) return;

    const std::vector<double> k = make_index_vector(states.size());

    const std::vector<double> x_hist   = get_col(states, 0);
    const std::vector<double> y_hist   = get_col(states, 1);
    const std::vector<double> psi_hist = get_col(states, 2);
    const std::vector<double> s_hist   = get_col(states, 3);

    std::vector<double> v_hist(states.size(), 0.0);
    bool has_v_state = !states.empty() && states[0].size() > 4;
    if (has_v_state) {
        v_hist = get_col(states, 4);
    }

    std::vector<double> v_ref_plot = v_ref;
    if (v_ref_plot.size() != states.size()) {
        if (v_ref_plot.empty()) {
            v_ref_plot = std::vector<double>(states.size(), 0.0);
        } else if (v_ref_plot.size() < states.size()) {
            double last = v_ref_plot.back();
            v_ref_plot.resize(states.size(), last);
        } else {
            v_ref_plot.resize(states.size());
        }
    }

    plt::figure_size(1400, 900);

    plt::subplot(3, 2, 1);
    plt::plot(k, x_hist);
    plt::title("x");
    plt::xlabel("step");
    plt::ylabel("x [m]");
    plt::grid(true);

    plt::subplot(3, 2, 2);
    plt::plot(k, y_hist);
    plt::title("y");
    plt::xlabel("step");
    plt::ylabel("y [m]");
    plt::grid(true);

    plt::subplot(3, 2, 3);
    plt::plot(k, psi_hist);
    plt::title("psi");
    plt::xlabel("step");
    plt::ylabel("psi [rad]");
    plt::grid(true);

    plt::subplot(3, 2, 4);
    plt::plot(k, s_hist);
    plt::title("s");
    plt::xlabel("step");
    plt::ylabel("s [m]");
    plt::grid(true);

    plt::subplot(3, 2, 5);
    if (has_v_state) {
        plt::named_plot("v", k, v_hist);
        plt::named_plot("v_ref", k, v_ref_plot, "--");
        plt::legend();
        plt::title("speed state");
        plt::xlabel("step");
        plt::ylabel("v [m/s]");
        plt::grid(true);
    } else {
        plt::plot(k, v_ref_plot, "--");
        plt::title("v_ref");
        plt::xlabel("step");
        plt::ylabel("v_ref [m/s]");
        plt::grid(true);
    }

    plt::subplot(3, 2, 6);
    plt::plot(x_hist, y_hist);
    plt::title("XY trajectory");
    plt::xlabel("x [m]");
    plt::ylabel("y [m]");
    plt::axis("equal");
    plt::grid(true);

    plt::tight_layout();
    plt::save(filename);
    plt::close();
}

void plot_inputs(const std::vector<std::vector<double>>& ctrls,
                 const std::vector<std::vector<double>>& states,
                 double theta_min, double theta_max,
                 double acc_min, double acc_max,
                 const std::string& filename)
{
    if (ctrls.empty()) return;

    const std::vector<double> k = make_index_vector(ctrls.size());

    const std::vector<double> v_cmd     = get_col(ctrls, 0);
    const std::vector<double> theta_cmd = get_col(ctrls, 1);
    const std::vector<double> p_cmd     = get_col(ctrls, 2);

    std::vector<double> theta_min_line = constant_vec(ctrls.size(), theta_min);
    std::vector<double> theta_max_line = constant_vec(ctrls.size(), theta_max);

    // Approximate longitudinal acceleration from velocity state if available
    std::vector<double> a_est(ctrls.size(), 0.0);
    bool has_v_state = !states.empty() && states[0].size() > 4 && states.size() >= ctrls.size() + 1;
    if (has_v_state) {
        for (std::size_t i = 0; i < ctrls.size(); ++i) {
            a_est[i] = states[i + 1][4] - states[i][4];
        }
    }

    std::vector<double> acc_min_line = constant_vec(ctrls.size(), acc_min);
    std::vector<double> acc_max_line = constant_vec(ctrls.size(), acc_max);

    plt::figure_size(1400, 900);

    plt::subplot(4, 1, 1);
    plt::plot(k, v_cmd);
    plt::title("input v");
    plt::xlabel("step");
    plt::ylabel("v_cmd [m/s]");
    plt::grid(true);

    plt::subplot(4, 1, 2);
    plt::named_plot("theta", k, theta_cmd);
    plt::named_plot("theta_min", k, theta_min_line, "--");
    plt::named_plot("theta_max", k, theta_max_line, "--");
    plt::legend();
    plt::title("input theta");
    plt::xlabel("step");
    plt::ylabel("theta [rad]");
    plt::grid(true);

    plt::subplot(4, 1, 3);
    plt::plot(k, p_cmd);
    plt::title("input p");
    plt::xlabel("step");
    plt::ylabel("p");
    plt::grid(true);

    plt::subplot(4, 1, 4);
    if (has_v_state) {
        plt::named_plot("a_est = v(k+1)-v(k)", k, a_est);
        plt::named_plot("acc_min", k, acc_min_line, "--");
        plt::named_plot("acc_max", k, acc_max_line, "--");
        plt::legend();
        plt::title("approx. acceleration");
        plt::xlabel("step");
        plt::ylabel("a_est [m/s per step]");
        plt::grid(true);
    } else {
        plt::plot(k, constant_vec(ctrls.size(), 0.0));
        plt::title("approx. acceleration unavailable");
        plt::xlabel("step");
        plt::ylabel("a_est");
        plt::grid(true);
    }

    plt::tight_layout();
    plt::save(filename);
    plt::close();
}

void plot_corridor(const std::vector<std::vector<double>>& states,
                   const std::vector<std::vector<double>>& left,
                   const std::vector<std::vector<double>>& right,
                   const std::vector<std::vector<ObstacleSpec>>& obs_log,
                   const std::string& filename)
{
    if (states.empty()) return;

    const std::vector<double> x_hist = get_col(states, 0);
    const std::vector<double> y_hist = get_col(states, 1);

    std::vector<double> left_x, left_y, right_x, right_y;
    left_x.reserve(left.size());
    left_y.reserve(left.size());
    right_x.reserve(right.size());
    right_y.reserve(right.size());

    for (const auto& p : left) {
        if (p.size() >= 2) {
            left_x.push_back(p[0]);
            left_y.push_back(p[1]);
        }
    }
    for (const auto& p : right) {
        if (p.size() >= 2) {
            right_x.push_back(p[0]);
            right_y.push_back(p[1]);
        }
    }

    plt::figure_size(1200, 900);

    // track boundaries
    if (!left_x.empty())  plt::named_plot("left", left_x, left_y);
    if (!right_x.empty()) plt::named_plot("right", right_x, right_y);

    // driven trajectory
    plt::named_plot("trajectory", x_hist, y_hist);

    // start / end markers
    if (!x_hist.empty()) {
        plt::scatter(std::vector<double>{x_hist.front()}, std::vector<double>{y_hist.front()}, 40.0);
        plt::scatter(std::vector<double>{x_hist.back()},  std::vector<double>{y_hist.back()},  40.0);
    }

    // obstacles: plot last available set
    if (!obs_log.empty() && !obs_log.back().empty()) {
        for (const auto& obs : obs_log.back()) {
            const int N = 80;
            std::vector<double> ox(N), oy(N);
            for (int i = 0; i < N; ++i) {
                double a = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(N - 1);
                ox[i] = obs.x + obs.radius * std::cos(a);
                oy[i] = obs.y + obs.radius * std::sin(a);
            }
            plt::plot(ox, oy);
        }
    }

    plt::title("corridor and driven trajectory");
    plt::xlabel("x [m]");
    plt::ylabel("y [m]");
    plt::axis("equal");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save(filename);
    plt::close();
}

void plot_timing(const std::vector<double>& t_cpu_s,
                 double dT,
                 const std::string& filename)
{
    if (t_cpu_s.empty()) return;

    const std::vector<double> k = make_index_vector(t_cpu_s.size());

    double mean = std::accumulate(t_cpu_s.begin(), t_cpu_s.end(), 0.0)
                / static_cast<double>(t_cpu_s.size());

    std::vector<double> mean_line = constant_vec(t_cpu_s.size(), mean);
    std::vector<double> dt_line   = constant_vec(t_cpu_s.size(), dT);

    plt::figure_size(1200, 700);
    plt::named_plot("solve time", k, t_cpu_s);
    plt::named_plot("mean", k, mean_line, "--");
    plt::named_plot("dT", k, dt_line, ":");
    plt::title("MPC solve time");
    plt::xlabel("step");
    plt::ylabel("time [s]");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save(filename);
    plt::close();
}

void plot_progress(const std::vector<std::vector<double>>& states,
                   const std::vector<std::vector<double>>& ctrls,
                   const std::string& filename)
{
    if (states.empty()) return;

    const std::vector<double> ks = make_index_vector(states.size());
    const std::vector<double> ku = make_index_vector(ctrls.size());

    const std::vector<double> s_hist = get_col(states, 3);
    const std::vector<double> p_hist = get_col(ctrls, 2);

    plt::figure_size(1200, 800);

    plt::subplot(2, 1, 1);
    plt::plot(ks, s_hist);
    plt::title("progress state");
    plt::xlabel("step");
    plt::ylabel("s [m]");
    plt::grid(true);

    plt::subplot(2, 1, 2);
    plt::plot(ku, p_hist);
    plt::title("progress input");
    plt::xlabel("step");
    plt::ylabel("p");
    plt::grid(true);

    plt::tight_layout();
    plt::save(filename);
    plt::close();
}

void plot_cost(const std::vector<double>& cost_hist,
               const std::string& filename)
{
    if (cost_hist.empty()) return;

    const std::vector<double> k = make_index_vector(cost_hist.size());

    plt::figure_size(1200, 700);
    plt::plot(k, cost_hist);
    plt::title("MPC objective value");
    plt::xlabel("step");
    plt::ylabel("cost");
    plt::grid(true);
    plt::tight_layout();
    plt::save(filename);
    plt::close();
}

void plot_tracking_errors(const std::vector<double>& cte_hist,
                          const std::string& filename)
{
    if (cte_hist.empty()) return;

    const std::vector<double> k = make_index_vector(cte_hist.size());

    plt::figure_size(1200, 700);
    plt::plot(k, cte_hist);
    plt::title("cross-track error");
    plt::xlabel("step");
    plt::ylabel("cte [m]");
    plt::grid(true);
    plt::tight_layout();
    plt::save(filename);
    plt::close();
}