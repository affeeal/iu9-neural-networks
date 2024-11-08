#pragma once

#include <Eigen/Dense>

namespace nn {

class IActivationFunction {
 public:
  virtual ~IActivationFunction() = default;

 public:
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& z) = 0;
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& z) = 0;
};

class Linear final : public IActivationFunction {
 public:
  Eigen::VectorXd Apply(const Eigen::VectorXd& z) override { return z; }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& z) override {
    return Eigen::MatrixXd::Identity(z.rows(), z.cols());
  }
};

class ReLU final : public IActivationFunction {
 public:
  Eigen::VectorXd Apply(const Eigen::VectorXd& z) override {
    return z.array().max(0.0);
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& z) override {
    return z.array().cwiseTypedGreaterOrEqual(0.0).matrix().asDiagonal();
  }
};

class LeakyReLU final : public IActivationFunction {
  std::function<double(double)> f_, f_prime_;

 public:
  LeakyReLU(const double alpha)
      : f_([alpha](const double x) { return x >= 0 ? x : alpha * x; }),
        f_prime_([alpha](const double x) { return x >= 0 ? 1 : alpha; }) {}

  Eigen::VectorXd Apply(const Eigen::VectorXd& z) override {
    return z.unaryExpr(f_);
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& z) override {
    return z.unaryExpr(f_prime_).asDiagonal();
  }
};

class Sigmoid final : public IActivationFunction {
 public:
  Eigen::VectorXd Apply(const Eigen::VectorXd& z) override {
    return 1.0 / (1.0 + (-z).array().exp());
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& z) override {
    const auto sigmoid = Apply(z);
    return (sigmoid.array() * (1 - sigmoid.array())).matrix().asDiagonal();
  }
};

class Tanh final : public IActivationFunction {
 public:
  Eigen::VectorXd Apply(const Eigen::VectorXd& z) override {
    const auto e_z = z.array().exp();
    const auto e_neg_z = (-z).array().exp();
    return (e_z - e_neg_z) / (e_z + e_neg_z);
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& z) override {
    const auto tanh = Apply(z);
    return (1 - tanh.array().square()).matrix().asDiagonal();
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
    return softmax.asDiagonal().toDenseMatrix() - softmax * softmax.transpose();
  }
};

}  // namespace nn
