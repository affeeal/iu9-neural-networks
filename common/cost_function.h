#pragma once

#include <Eigen/Dense>
#include <cmath>

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
 public:
  double Apply(const Eigen::VectorXd& y, const Eigen::VectorXd& a) override {
    return 0.5 * (y - a).squaredNorm();
  }

  Eigen::VectorXd GradientWrtActivations(const Eigen::VectorXd& y,
                                         const Eigen::VectorXd& a) override {
    return a - y;
  }
};

class CrossEntropy final : public ICostFunction {
 public:
  double Apply(const Eigen::VectorXd& y, const Eigen::VectorXd& a) override {
    double result = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
      if (y[i] != 0) result -= y[i] * std::log(a[i]);
    }
    return result;
  }

  Eigen::VectorXd GradientWrtActivations(const Eigen::VectorXd& y,
                                         const Eigen::VectorXd& a) override {
    Eigen::VectorXd result(y.size());
    for (Eigen::Index i = 0; i < y.size(); ++i) {
      result[i] = y[i] == 0 ? 0 : -y[i] / a[i];
    }
    return result;
  }
};

class KLDivergence final : public ICostFunction {
 public:
  double Apply(const Eigen::VectorXd& y, const Eigen::VectorXd& a) override {
    double result = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
      if (y[i] != 0) result += y[i] * (std::log(y[i]) - std::log(a[i]));
    }
    return result;
  }

  Eigen::VectorXd GradientWrtActivations(const Eigen::VectorXd& y,
                                         const Eigen::VectorXd& a) override {
    Eigen::VectorXd result(y.size());
    for (Eigen::Index i = 0; i < y.size(); ++i) {
      result[i] = y[i] == 0 ? 0 : -y[i] / a[i];
    }
    return result;
  }
};

}  // namespace nn
