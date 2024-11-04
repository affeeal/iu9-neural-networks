#pragma once

// clang-format off
#include <Eigen/Dense>
// clang-format on

namespace nn {

class IActivationFunction {
 public:
  virtual ~IActivationFunction() = default;

 public:
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& z) = 0;
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& z) = 0;
};

class Sigmoid final : public IActivationFunction {
 public:
  Eigen::VectorXd Apply(const Eigen::VectorXd& z) override {
    return 1.0 / (1.0 + (-z).array().exp());
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& z) override {
    const auto sigmoid = Apply(z);
    return static_cast<Eigen::VectorXd>(sigmoid.array() * (1 - sigmoid.array())).asDiagonal();
  }
};

class Tanh final : public IActivationFunction {
  Eigen::VectorXd Apply(const Eigen::VectorXd& z) override {
    const auto e_z = z.array().exp();
    const auto e_neg_z = (-z).array().exp();
    return (e_z - e_neg_z) / (e_z + e_neg_z);
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& z) override {
    const auto tanh = Apply(z);
    return static_cast<Eigen::VectorXd>(1 - tanh.array().square()).asDiagonal();
  }
};

class Softmax final : public IActivationFunction {
 public:
  Eigen::VectorXd Apply(const Eigen::VectorXd& z) override {
    const auto e_z = z.array().exp();
    return e_z / e_z.sum();
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& z) override {
    const auto softmax = Apply(z);
    return static_cast<Eigen::MatrixXd>(softmax.asDiagonal()) - softmax * softmax.transpose();
  }
};

}  // namespace nn
