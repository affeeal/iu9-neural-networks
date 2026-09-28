#include "optimization.h"

#include <spdlog/spdlog.h>

#include "check.h"

namespace {
class Quadratic final : public hw3::IMultivariateFunction {
 public:
  std::size_t Size() const override { return 2; }
  double At(const Eigen::VectorXd& x) const override {
    const Eigen::Vector2d d = x - Eigen::Vector2d(1, -2);
    return 0.5 * d.dot(Hessian(x) * d);
  }
  Eigen::VectorXd Gradient(const Eigen::VectorXd& x) const override {
    return Hessian(x) * (x - Eigen::Vector2d(1, -2));
  }
  Eigen::MatrixXd Hessian(const Eigen::VectorXd&) const override {
    return Eigen::Matrix2d{{4, 1}, {1, 2}};
  }
};

class DoubleWell final : public hw3::IMultivariateFunction {
 public:
  std::size_t Size() const override { return 2; }
  double At(const Eigen::VectorXd& x) const override {
    return 1e6 * std::pow(x[0] * x[0] - 1, 2) + x[1] * x[1];
  }
  Eigen::VectorXd Gradient(const Eigen::VectorXd& x) const override {
    return Eigen::Vector2d(4e6 * x[0] * (x[0] * x[0] - 1), 2 * x[1]);
  }
  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const override {
    return Eigen::Matrix2d{{4e6 * (3 * x[0] * x[0] - 1), 0}, {0, 2}};
  }
};
}  // namespace

int main() {
  spdlog::set_level(spdlog::level::off);
  const auto parabola = [](double x) { return (x - 2.3) * (x - 2.3); };
  Near(hw3::FibonacciSearch(parabola, -10, 10), 2.3, 1e-7);
  Near(hw3::FibonacciSearch(parabola, 4, 10), 4, 1e-7);
  Throws([&] { hw3::FibonacciSearch(parabola, 10, -10); });
  const hw3::RosenbrockFunction rosenbrock;
  const Eigen::Vector2d point(-1.2, 1);
  const auto gradient = rosenbrock.Gradient(point);
  const auto hessian = rosenbrock.Hessian(point);
  constexpr double h = 1e-5;
  for (int i = 0; i < 2; ++i) {
    auto plus = point, minus = point;
    plus[i] += h;
    minus[i] -= h;
    Near(gradient[i], (rosenbrock.At(plus) - rosenbrock.At(minus)) / (2 * h),
         1e-5);
    const Eigen::VectorXd numeric =
        (rosenbrock.Gradient(plus) - rosenbrock.Gradient(minus)) / (2 * h);
    Near((hessian.col(i) - numeric).norm(), 0, 1e-5);
  }
  const Quadratic quadratic;
  const Eigen::Vector2d start(5, 5), optimum(1, -2);
  for (auto method : {hw3::GradientDescent, hw3::FletcherReeves,
                      hw3::PolakRibier, hw3::DavidonFletcherPowell}) {
    auto result = method(quadratic, start, 100, 1e-9, 1e-12, 1e-12, 0, 2);
    Near((result - optimum).norm(), 0, 1e-5);
    result = method(rosenbrock, point, 100, 1e-9, 1e-12, 1e-12, 0, 1);
    CHECK(result.allFinite() && rosenbrock.At(result) < rosenbrock.At(point));
    Throws([&] {
      method(quadratic, Eigen::VectorXd{}, 10, 1e-9, 1e-9, 1e-9, 0, 1);
    });
  }
  const auto lm = hw3::LevenbergMarquardt(quadratic, start, 100, 1e-9);
  Near((lm - optimum).norm(), 0, 1e-5);
  const Eigen::Vector2d reject_start(0.1, 0);
  const auto rejected =
      hw3::LevenbergMarquardt(DoubleWell{}, reject_start, 1, 1e-9);
  Near((rejected - reject_start).norm(), 0);
}
