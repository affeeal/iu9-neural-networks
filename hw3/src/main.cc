#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <eigen3/Eigen/Core>

namespace {

class IFunction {
 public:
  virtual ~IFunction() = default;

 public:
  virtual std::size_t Size() const = 0;

  virtual double Calculate(const Eigen::VectorXd& u) const = 0;

  virtual Eigen::VectorXd Gradient(const Eigen::VectorXd& u) const = 0;

  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& u) const = 0;
};

class RosenbrockFunction final : public IFunction {
 public:
  static constexpr std::size_t kInputSize = 2;

  std::size_t Size() const override { return kInputSize; }

  double Calculate(const Eigen::VectorXd& u) const override {
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

class Paraboloid final : public IFunction {
 public:
  static constexpr std::size_t kInputSize = 2;

  std::size_t Size() const override { return kInputSize; }

  double Calculate(const Eigen::VectorXd& u) const override {
    return std::pow(u.x() - 1, 2) + std::pow(u.y() - 1, 2);
  }

  Eigen::VectorXd Gradient(const Eigen::VectorXd& u) const override {
    return Eigen::Vector<double, kInputSize>{2 * (u.x() - 1), 2 * (u.y() - 1)};
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& u) const override {
    return Eigen::Matrix<double, kInputSize, kInputSize>{
        {2, 0},
        {0, 2},
    };
  }
};

struct Config final {
  double a, b;
  double delta, epsilon;
};

double DichotomyMinimum(const std::function<double(double)>& f,
                        const Config& cfg) {
  auto a = cfg.a;
  auto b = cfg.b;

  while (b - a >= cfg.epsilon) {
    const auto x1 = 0.5 * (a + b) - cfg.delta;
    const auto x2 = 0.5 * (a + b) + cfg.delta;

    const auto f1 = f(x1);
    const auto f2 = f(x2);

    if (f1 < f2) {
      b = x1;
    } else if (f1 > f2) {
      a = x2;
    } else {
      a = x1;
      b = x2;
    }
  }

  return 0.5 * (a + b);
}

Eigen::VectorXd FletcherReeves(const IFunction& f, const Eigen::VectorXd& x0,
                               const std::size_t max_iterations,
                               const double grad_epsilon, const double delta,
                               const double epsilon, const Config& cfg) {
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd prev_grad, grad;
  Eigen::VectorXd prev_d, d;
  const auto phi = [&](const double alpha) {
    return f.Calculate(x + alpha * d);
  };
  for (std::size_t k = 0; k < max_iterations; ++k) {
    grad = f.Gradient(x);
    if (grad.norm() < grad_epsilon) {
      return x;
    }

    d = -grad;
    if (k > 0) {
      const auto w_prev = grad.squaredNorm() / prev_grad.squaredNorm();
      d += w_prev * prev_grad;
    }

    const auto alpha = DichotomyMinimum(phi, cfg);
    x_next = x + alpha * d;

    if ((x_next - x).norm() < delta &&
        std::abs(f.Calculate(x_next) - f.Calculate(x)) < epsilon) {
      return x_next;
    }

    x = std::move(x_next);
    prev_grad = std::move(grad);
    prev_d = std::move(d);
  }

  return x;
}

Eigen::VectorXd PolakRibier(const IFunction& f, const Eigen::VectorXd& x0,
                            const std::size_t max_iterations,
                            const double grad_epsilon, const double delta,
                            const double epsilon, const Config& cfg) {
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd prev_grad, grad;
  Eigen::VectorXd prev_d, d;
  double alpha;
  const auto phi = [&](const double alpha) {
    return f.Calculate(x + alpha * d);
  };
  const auto n = f.Size();
  for (std::size_t k = 0; k < max_iterations; ++k) {
    grad = f.Gradient(x);
    if (grad.norm() < grad_epsilon) {
      return x;
    }

    d = -grad;
    if (k > 0) {
      const auto w_prev =
          (k % n == 0 ? 0
                      : grad.dot(grad - prev_grad) / prev_grad.squaredNorm());
      d += w_prev * prev_grad;
    }

    alpha = DichotomyMinimum(phi, cfg);
    x_next = x + alpha * d;

    if ((x_next - x).norm() < delta &&
        std::abs(f.Calculate(x_next) - f.Calculate(x)) < epsilon) {
      return x_next;
    }

    x = std::move(x_next);
    prev_grad = std::move(grad);
    prev_d = std::move(d);
  }

  return x;
}

Eigen::VectorXd DavidonFletcherPowell(const IFunction& f,
                                      const Eigen::VectorXd& x0,
                                      const std::size_t max_iterations,
                                      const double grad_epsilon,
                                      const double delta, const double epsilon,
                                      const Config& cfg) {
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd grad = f.Gradient(x), grad_next;
  Eigen::VectorXd d;

  const auto phi = [&](const double alpha) {
    return f.Calculate(x + alpha * d);
  };

  const auto n = f.Size();
  Eigen::MatrixXd g(n, n);
  g.setIdentity();

  for (std::size_t k = 0; k < max_iterations; ++k) {
    if (grad.norm() < grad_epsilon) {
      return x;
    }

    d = -g * grad;
    const auto alpha = DichotomyMinimum(phi, cfg);
    x_next = x + alpha * d;

    if ((x_next - x).norm() < delta &&
        std::abs(f.Calculate(x_next) - f.Calculate(x)) < epsilon) {
      return x_next;
    }

    const Eigen::VectorXd delta_x = x_next - x;
    grad_next = f.Gradient(x_next);
    const Eigen::VectorXd delta_grad = grad_next - grad;
    const Eigen::VectorXd w1 = delta_x;
    const Eigen::VectorXd w2 = g * delta_grad;
    const double sigma1 = 1 / w1.dot(delta_grad);
    const double sigma2 = -1 / w2.dot(delta_grad);
    g += sigma1 * w1 * w1.transpose() + sigma2 * w2 * w2.transpose();

    x = std::move(x_next);
    grad = std::move(grad_next);
  }

  return x;
}

Eigen::VectorXd LevenbergMarquardt(const IFunction& f,
                                   const Eigen::VectorXd& x0,
                                   const std::size_t max_iterations,
                                   const double epsilon) {
  auto x = x0;
  auto mu = 1e+4;
  const auto n = f.Size();

  for (std::size_t k = 0; k < max_iterations; ++k) {
    const auto grad = f.Gradient(x);
    if (grad.norm() < epsilon) {
      return x;
    }

    const auto f_x = f.Calculate(x);
    x -= (f.Hessian(x) + mu * Eigen::MatrixXd::Identity(n, n)).inverse() * grad;
    if (f.Calculate(x) < f_x) {
      mu /= 2;
    } else {
      mu *= 2;
    }
  }

  return x;
}

}  // namespace

int main() {
  spdlog::set_level(spdlog::level::level_enum::debug);

  const auto f = RosenbrockFunction();
  const auto x0 = Eigen::Vector<double, 2>{100, 100};
  const auto cfg =
      Config{.a = -10.0, .b = 10.0, .delta = 1e-10, .epsilon = 1e-9};

  constexpr auto kDelta = 1e-10;
  constexpr auto kEpsilon = 1e-9;
  constexpr auto kGradientEpsilon = 1e-9;
  constexpr std::size_t kMaxIterations = 100;

  auto u = FletcherReeves(f, x0, kMaxIterations, kGradientEpsilon, kDelta,
                          kEpsilon, cfg);
  spdlog::info("Fletcher-Reeves: x=({}, {}), f(x)={}", u.x(), u.y(),
               f.Calculate(u));

  u = PolakRibier(f, x0, kMaxIterations, kGradientEpsilon, kDelta, kEpsilon,
                  cfg);
  spdlog::info("Polak-Ribier: x=({}, {}), f(x)={}", u.x(), u.y(),
               f.Calculate(u));

  u = DavidonFletcherPowell(f, x0, kMaxIterations, kGradientEpsilon, kDelta,
                            kEpsilon, cfg);
  spdlog::info("Davidon-Fletcher-Powell: x=({}, {}), f(x)={}", u.x(), u.y(),
               f.Calculate(u));

  u = LevenbergMarquardt(f, x0, kMaxIterations, kGradientEpsilon);
  spdlog::info("Levenberg-Marquardt: x=({}, {}), f(x)={}", u.x(), u.y(),
               f.Calculate(u));
}
