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
    // std::cout << "?\n";
    return 1.0 / (1.0 + z.array().inverse().exp());
  }

  Eigen::VectorXd Prime(const Eigen::VectorXd& z) override {
    // std::cout << "???\n";
    return Apply(z).array() * (1 - Apply(z).array());
  }
};

}  // namespace lab1
