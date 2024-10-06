#pragma once

#include <Eigen/Dense>
#include <cmath>

#include "activation_function.h"

namespace lab1 {

class ICostFunction {
 public:
  virtual ~ICostFunction() = default;

 public:
  virtual double Apply(const Eigen::VectorXd& y, const Eigen::VectorXd& a) = 0;
  virtual Eigen::VectorXd PrimeActivations(const Eigen::VectorXd& y,
                                           const Eigen::VectorXd& a) = 0;
};

class MSE final : public ICostFunction {
  double Apply(const Eigen::VectorXd& y, const Eigen::VectorXd& a) override {
    return 0.5 * std::pow((y - a).norm(), 2);
  }

  Eigen::VectorXd PrimeActivations(const Eigen::VectorXd& y,
                                   const Eigen::VectorXd& a) override {
    return a - y;
  }
};

}  // namespace lab1
