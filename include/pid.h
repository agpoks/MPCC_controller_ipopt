#pragma once
#include <algorithm>

// Simple discrete-time PID with optional anti-windup and output clamping.
// Used as an outer loop to convert progress-rate command to acceleration.
class PIDSpeed {
public:
    PIDSpeed(double kp, double ki, double kd, double dt,
             double out_min, double out_max,
             bool anti_windup = true, bool clamp_output = true)
        : kp_(kp), ki_(ki), kd_(kd), dt_(dt),
          out_min_(out_min), out_max_(out_max),
          anti_windup_(anti_windup), clamp_output_(clamp_output),
          integral_(0.0), prev_error_(0.0) {}

    double step(double setpoint, double measured) {
        double error = setpoint - measured;
        integral_ += ki_ * error * dt_;
        if (anti_windup_)
            integral_ = std::clamp(integral_, out_min_, out_max_);
        double derivative = (error - prev_error_) / dt_;
        prev_error_ = error;
        double out = kp_ * error + integral_ + kd_ * derivative;
        if (clamp_output_)
            out = std::clamp(out, out_min_, out_max_);
        return out;
    }

    void reset() { integral_ = 0.0; prev_error_ = 0.0; }

private:
    double kp_, ki_, kd_, dt_;
    double out_min_, out_max_;
    bool anti_windup_, clamp_output_;
    double integral_, prev_error_;
};
