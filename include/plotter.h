//
// Created by poxx on 3/12/26.
//

#ifndef PLOTTER_H
#define PLOTTER_H
#pragma once
#include <vector>
#include <string>
#pragma once

#include <string>
#include <vector>

#include "utils.h"
#include "MPCC_planner.h"

// states: rows like [x, y, psi, s] or [x, y, psi, s, v]
// ctrls : rows like [v, theta, p]

void plot_outputs(const std::vector<std::vector<double>>& states,
                  const std::vector<double>& v_ref,
                  const std::string& filename);

void plot_inputs(const std::vector<std::vector<double>>& ctrls,
                 const std::vector<std::vector<double>>& states,
                 double theta_min, double theta_max,
                 double acc_min, double acc_max,
                 const std::string& filename);

void plot_corridor(const std::vector<std::vector<double>>& states,
                   const std::vector<std::vector<double>>& left,
                   const std::vector<std::vector<double>>& right,
                   const std::vector<std::vector<ObstacleSpec>>& obs_log,
                   const std::string& filename);

void plot_timing(const std::vector<double>& t_cpu_s,
                 double dT,
                 const std::string& filename);

void plot_progress(const std::vector<std::vector<double>>& states,
                   const std::vector<std::vector<double>>& ctrls,
                   const std::string& filename);

void plot_cost(const std::vector<double>& cost_hist,
               const std::string& filename);

void plot_tracking_errors(const std::vector<double>& cte_hist,
                          const std::string& filename);
#endif //PLOTTER_H
