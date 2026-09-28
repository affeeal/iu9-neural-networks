#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <functional>

namespace hw3 {

class IMultivariateFunction {
 public:
  virtual ~IMultivariateFunction() = default;

 public:
  virtual std::size_t Size() const = 0;

  virtual double At(const Eigen::VectorXd& u) const = 0;

  virtual Eigen::VectorXd Gradient(const Eigen::VectorXd& u) const = 0;

  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& u) const = 0;
};

class RosenbrockFunction final : public IMultivariateFunction {
 public:
  static constexpr std::size_t kInputSize = 2;

  std::size_t Size() const override { return kInputSize; }

  double At(const Eigen::VectorXd& u) const override {
    return 250 * std::pow(std::pow(u.x(), 2) - u.y(), 2) +
           2 * std::pow(u.x() - 1, 2) + 50;
  }

  Eigen::VectorXd Gradient(const Eigen::VectorXd& u) const override {
    return Eigen::Vector<double, kInputSize>{
        1000 * std::pow(u.x(), 3) - 1000 * u.x() * u.y() + 4 * u.x() - 4,
        -500 * std::pow(u.x(), 2) + 500 * u.y()};
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& u) const override {
    return Eigen::Matrix<double, kInputSize, kInputSize>{
        {3000 * std::pow(u.x(), 2) - 1000 * u.y() + 4, -1000 * u.x()},
        {-1000 * u.x(), 500},
    };
  }
};

Eigen::VectorXd GradientDescent(const IMultivariateFunction& f,
                                const Eigen::VectorXd& x0,
                                const std::size_t max_iterations,
                                const double grad_epsilon, const double delta,
                                const double epsilon, const double a,
                                const double b);

Eigen::VectorXd FletcherReeves(const IMultivariateFunction& f,
                               const Eigen::VectorXd& x0,
                               const std::size_t max_iterations,
                               const double grad_epsilon, const double delta,
                               const double epsilon, const double a,
                               const double b);

Eigen::VectorXd PolakRibier(const IMultivariateFunction& f,
                            const Eigen::VectorXd& x0,
                            const std::size_t max_iterations,
                            const double grad_epsilon, const double delta,
                            const double epsilon, const double a,
                            const double b);

Eigen::VectorXd DavidonFletcherPowell(const IMultivariateFunction& f,
                                      const Eigen::VectorXd& x0,
                                      const std::size_t max_iterations,
                                      const double grad_epsilon,
                                      const double delta, const double epsilon,
                                      const double a, const double b);

Eigen::VectorXd LevenbergMarquardt(const IMultivariateFunction& f,
                                   const Eigen::VectorXd& x0,
                                   const std::size_t max_iterations,
                                   const double epsilon);

double FibonacciSearch(const std::function<double(double)>& f, double a,
                       double b);

}  // namespace hw3
