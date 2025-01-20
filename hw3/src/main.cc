#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <cmath>

namespace {

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

template <std::size_t N>
constexpr std::array<std::size_t, N + 1> GetFibonacciNumbers() {
  auto numbers = std::array<std::size_t, N + 1>{};
  numbers[0] = 0;
  numbers[1] = 1;
  for (std::size_t i = 2; i <= N; ++i) {
    numbers[i] = numbers[i - 2] + numbers[i - 1];
  }
  return numbers;
}

double FibonacciSearch(const std::function<double(double)>& f, const double a,
                       const double b) {
  static constexpr std::size_t kN = 150;
  static constexpr auto kF = GetFibonacciNumbers<kN>();

  auto xl = a;
  auto xr = b;
  auto l0 = xr - xl;
  auto li = static_cast<double>(kF[kN - 2]) / static_cast<double>(kF[kN]) * l0;

  double x1 = 0, x2 = 0, f1 = 0, f2 = 0;
  for (std::size_t i = 2; i <= kN; ++i) {
    if (li > l0 / 2) {
      x1 = xr - li;
      x2 = xl + li;
    } else {
      x1 = xl + li;
      x2 = xr - li;
    }

    f1 = f(x1);
    f2 = f(x2);

    if (f1 < f2) {
      xr = x2;
      li = static_cast<double>(kF[kN - i]) / kF[kN - (i - 2)] * l0;
    } else if (f1 > f2) {
      xl = x1;
      li = static_cast<double>(kF[kN - i]) / kF[kN - (i - 2)] * l0;
    } else {
      xl = x1;
      xr = x2;
      li = static_cast<double>(kF[kN - i]) / kF[kN - (i - 2)] * (xr - xl);
    }

    l0 = xr - xl;
  }

  if (f1 <= f2) {
    return x1;
  }
  return x2;
}

Eigen::VectorXd GradientDescent(const IMultivariateFunction& f,
                                const Eigen::VectorXd& x0,
                                const std::size_t max_iterations,
                                const double grad_epsilon, const double delta,
                                const double epsilon, const double a,
                                const double b) {
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd grad;
  const auto phi = [&](const double alpha) {
    return f.At(x - alpha * grad);
  };
  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::debug("Iteration {}, x = ({}, {}), f(x)={}", k, x.x(), x.y(), f.At(x));
    grad = f.Gradient(x);
    if (grad.norm() < grad_epsilon) {
      spdlog::debug("||grad|| < epsilon");
      return x;
    }

    const auto alpha = FibonacciSearch(phi, a, b);
    x_next = x - alpha * grad;

    if ((x_next - x).norm() < delta &&
        std::abs(f.At(x_next) - f.At(x)) < epsilon) {
      spdlog::debug("||x_next - x|| < delta && |f(x_next) - f(x)| < epsilon");
      return x_next;
    }

    x = std::move(x_next);
  }

  return x;
}

Eigen::VectorXd FletcherReeves(const IMultivariateFunction& f,
                               const Eigen::VectorXd& x0,
                               const std::size_t max_iterations,
                               const double grad_epsilon, const double delta,
                               const double epsilon, const double a,
                               const double b) {
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd prev_grad, grad;
  Eigen::VectorXd prev_d, d;
  const auto phi = [&](const double alpha) {
    return f.At(x + alpha * d);
  };
  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::debug("Iteration {}, x = ({}, {}), f(x)={}", k, x.x(), x.y(), f.At(x));
    grad = f.Gradient(x);
    if (grad.norm() < grad_epsilon) {
      spdlog::debug("||grad|| < epsilon");
      return x;
    }

    d = -grad;
    if (k > 0) {
      const auto w_prev = grad.squaredNorm() / prev_grad.squaredNorm();
      d += w_prev * prev_grad;
    }

    const auto alpha = FibonacciSearch(phi, a, b);
    x_next = x + alpha * d;

    if ((x_next - x).norm() < delta &&
        std::abs(f.At(x_next) - f.At(x)) < epsilon) {
      spdlog::debug("||x_next - x|| < delta && |f(x_next) - f(x)| < epsilon");
      return x_next;
    }

    x = std::move(x_next);
    prev_grad = std::move(grad);
    prev_d = std::move(d);
  }

  return x;
}

Eigen::VectorXd PolakRibier(const IMultivariateFunction& f,
                            const Eigen::VectorXd& x0,
                            const std::size_t max_iterations,
                            const double grad_epsilon, const double delta,
                            const double epsilon, const double a,
                            const double b) {
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd prev_grad, grad;
  Eigen::VectorXd prev_d, d;
  double alpha;
  const auto phi = [&](const double alpha) {
    return f.At(x + alpha * d);
  };
  const auto n = f.Size();
  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::debug("Iteration {}, x = ({}, {}), f(x)={}", k, x.x(), x.y(), f.At(x));
    grad = f.Gradient(x);
    if (grad.norm() < grad_epsilon) {
      spdlog::debug("||grad|| < epsilon");
      return x;
    }

    d = -grad;
    if (k > 0) {
      const auto w_prev =
          (k % n == 0 ? 0
                      : grad.dot(grad - prev_grad) / prev_grad.squaredNorm());
      d += w_prev * prev_grad;
    }

    alpha = FibonacciSearch(phi, a, b);
    x_next = x + alpha * d;

    if ((x_next - x).norm() < delta &&
        std::abs(f.At(x_next) - f.At(x)) < epsilon) {
      spdlog::debug("||x_next - x|| < delta && |f(x_next) - f(x)| < epsilon");
      return x_next;
    }

    x = std::move(x_next);
    prev_grad = std::move(grad);
    prev_d = std::move(d);
  }

  return x;
}

Eigen::VectorXd DavidonFletcherPowell(const IMultivariateFunction& f,
                                      const Eigen::VectorXd& x0,
                                      const std::size_t max_iterations,
                                      const double grad_epsilon,
                                      const double delta, const double epsilon,
                                      const double a, const double b) {
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd grad = f.Gradient(x), grad_next;
  Eigen::VectorXd d;

  const auto phi = [&](const double alpha) {
    return f.At(x + alpha * d);
  };

  const auto n = f.Size();
  Eigen::MatrixXd g(n, n);
  g.setIdentity();

  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::debug("Iteration {}, x = ({}, {}), f(x)={}", k, x.x(), x.y(), f.At(x));
    if (grad.norm() < grad_epsilon) {
      spdlog::debug("||grad|| < epsilon");
      return x;
    }

    d = -g * grad;
    const auto alpha = FibonacciSearch(phi, a, b);
    x_next = x + alpha * d;

    if ((x_next - x).norm() < delta &&
        std::abs(f.At(x_next) - f.At(x)) < epsilon) {
      spdlog::debug("||x_next - x|| < delta && |f(x_next) - f(x)| < epsilon");
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

Eigen::VectorXd LevenbergMarquardt(const IMultivariateFunction& f,
                                   const Eigen::VectorXd& x0,
                                   const std::size_t max_iterations,
                                   const double epsilon) {
  auto x = x0;
  auto mu = 1e+4;
  const auto n = f.Size();

  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::debug("Iteration {}, x = ({}, {}), f(x)={}", k, x.x(), x.y(), f.At(x));
    const auto grad = f.Gradient(x);
    if (grad.norm() < epsilon) {
      spdlog::debug("||grad|| < epsilon");
      return x;
    }

    const auto f_x = f.At(x);
    x -= (f.Hessian(x) + mu * Eigen::MatrixXd::Identity(n, n)).inverse() * grad;
    if (f.At(x) < f_x) {
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

  constexpr std::size_t kMaxIterations = 100;
  constexpr auto kDelta = 1e-10;
  constexpr auto kEpsilon = 1e-9;
  constexpr auto kGradientEpsilon = 1e-9;
  constexpr auto kA = -10.0;
  constexpr auto kB = 10.0;

  auto u = GradientDescent(f, x0, kMaxIterations, kGradientEpsilon, kDelta,
                           kEpsilon, kA, kB);
  spdlog::info("Gradient descent: x=({}, {}), f(x)={}", u.x(), u.y(),
               f.At(u));

  u = FletcherReeves(f, x0, kMaxIterations, kGradientEpsilon, kDelta, kEpsilon,
                     kA, kB);
  spdlog::info("Fletcher-Reeves: x=({}, {}), f(x)={}", u.x(), u.y(),
               f.At(u));

  u = PolakRibier(f, x0, kMaxIterations, kGradientEpsilon, kDelta, kEpsilon, kA,
                  kB);
  spdlog::info("Polak-Ribier: x=({}, {}), f(x)={}", u.x(), u.y(),
               f.At(u));

  u = DavidonFletcherPowell(f, x0, kMaxIterations, kGradientEpsilon, kDelta,
                            kEpsilon, kA, kB);
  spdlog::info("Davidon-Fletcher-Powell: x=({}, {}), f(x)={}", u.x(), u.y(),
               f.At(u));

  u = LevenbergMarquardt(f, x0, kMaxIterations, kGradientEpsilon);
  spdlog::info("Levenberg-Marquardt: x=({}, {}), f(x)={}", u.x(), u.y(),
               f.At(u));
}
