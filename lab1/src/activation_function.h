#pragma once

#include <Eigen/Dense>
#include <iostream>

namespace lab1 {

class IActivationFunction {
 public:
  virtual ~IActivationFunction() = default;

 public:
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& z) = 0;
  virtual Eigen::VectorXd Prime(const Eigen::VectorXd& z) = 0;
};

class Sigmoid final : public IActivationFunction {
 public:
  Eigen::VectorXd Apply(const Eigen::VectorXd& z) override {
    return 1.0 / (1.0 + (-z).array().exp());
  }

  Eigen::VectorXd Prime(const Eigen::VectorXd& z) override {
    const auto activation = Apply(z);
    return activation.array() * (1 - activation.array());
  }
};

class Tanh final : public IActivationFunction {
  Eigen::VectorXd Apply(const Eigen::VectorXd& z) override {
    const auto e_z = z.array().exp();
    const auto e_minus_z = (-z).array().exp();
    return (e_z - e_minus_z) / (e_z + e_minus_z);
  }

  Eigen::VectorXd Prime(const Eigen::VectorXd& z) override {
    const auto tanh = Apply(z);
    return 1 - tanh.array().square();
  }
};

/*
class Softmax final : public IActivationFunction {
 public:
  Eigen::VectorXd Apply(const Eigen::VectorXd& z) override {
    // TODO
  }

  Eigen::VectorXd Prime(const Eigen::VectorXd& z) override {
    // TODO
  }
};
*/

}  // namespace lab1
