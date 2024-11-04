#pragma once

#include "activation_function.h"

namespace nn {

class ICostFunction {
 public:
  virtual ~ICostFunction() = default;

 public:
  virtual double Apply(const Eigen::VectorXd& y, const Eigen::VectorXd& a) = 0;
  virtual Eigen::VectorXd GradientWrtActivations(const Eigen::VectorXd& y,
                                                 const Eigen::VectorXd& a) = 0;
};

class MSE final : public ICostFunction {
  double Apply(const Eigen::VectorXd& y, const Eigen::VectorXd& a) override {
    return 0.5 * (y - a).squaredNorm();
  }

  Eigen::VectorXd GradientWrtActivations(const Eigen::VectorXd& y,
                                         const Eigen::VectorXd& a) override {
    return a - y;
  }
};

class CrossEntropy final : public ICostFunction {
  double Apply(const Eigen::VectorXd& y, const Eigen::VectorXd& a) override {
    return -(y.array() * a.array().log()).sum();
  }

  Eigen::VectorXd GradientWrtActivations(const Eigen::VectorXd& y,
                                         const Eigen::VectorXd& a) override {
    return - y.array() / a.array();
  }
};

}  // namespace nn
