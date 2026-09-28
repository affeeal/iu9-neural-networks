#include "optimization.h"

#include <spdlog/spdlog.h>

#include <array>
#include <stdexcept>

namespace hw3 {

double FibonacciSearch(const std::function<double(double)>& f, double a,
                       double b) {
  if (!std::isfinite(a) || !std::isfinite(b) || a > b) {
    throw std::invalid_argument("Invalid line-search interval");
  }
  if (a == b) return a;
  // Floating-point Fibonacci numbers avoid size_t overflow at F(94).
  constexpr std::size_t n = 64;
  std::array<double, n + 1> fib{};
  fib[1] = 1;
  for (std::size_t i = 2; i <= n; ++i) fib[i] = fib[i - 1] + fib[i - 2];
  double x1 = a + fib[n - 2] / fib[n] * (b - a);
  double x2 = a + fib[n - 1] / fib[n] * (b - a);
  const auto evaluate = [&](double x) {
    const auto value = f(x);
    if (!std::isfinite(value))
      throw std::runtime_error("Nonfinite line-search value");
    return value;
  };
  double f1 = evaluate(x1), f2 = evaluate(x2);
  for (std::size_t k = 1; k < n - 2; ++k) {
    if (f1 > f2) {
      a = x1;
      x1 = x2;
      f1 = f2;
      x2 = a + fib[n - k - 1] / fib[n - k] * (b - a);
      f2 = evaluate(x2);
    } else {
      b = x2;
      x2 = x1;
      f2 = f1;
      x1 = a + fib[n - k - 2] / fib[n - k] * (b - a);
      f1 = evaluate(x1);
    }
  }
  return (a + b) / 2;
}

Eigen::VectorXd GradientDescent(const IMultivariateFunction& f,
                                const Eigen::VectorXd& x0,
                                const std::size_t max_iterations,
                                const double grad_epsilon, const double delta,
                                const double epsilon, const double a,
                                const double b) {
  if (f.Size() == 0 || x0.size() != static_cast<Eigen::Index>(f.Size()) ||
      !x0.allFinite()) {
    throw std::invalid_argument("Invalid initial point");
  }
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd grad;
  const auto phi = [&](const double alpha) { return f.At(x - alpha * grad); };
  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::debug("Iteration {}, f(x)={}", k, f.At(x));
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
  if (f.Size() == 0 || x0.size() != static_cast<Eigen::Index>(f.Size()) ||
      !x0.allFinite()) {
    throw std::invalid_argument("Invalid initial point");
  }
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd prev_grad, grad;
  Eigen::VectorXd prev_d, d;
  const auto phi = [&](const double alpha) { return f.At(x + alpha * d); };
  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::debug("Iteration {}, f(x)={}", k, f.At(x));
    grad = f.Gradient(x);
    if (grad.norm() < grad_epsilon) {
      spdlog::debug("||grad|| < epsilon");
      return x;
    }

    d = -grad;
    if (k > 0) {
      const auto w_prev = grad.squaredNorm() / prev_grad.squaredNorm();
      d += w_prev * prev_d;
      if (d.dot(grad) >= 0) d = -grad;
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
  if (f.Size() == 0 || x0.size() != static_cast<Eigen::Index>(f.Size()) ||
      !x0.allFinite()) {
    throw std::invalid_argument("Invalid initial point");
  }
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd prev_grad, grad;
  Eigen::VectorXd prev_d, d;
  double alpha;
  const auto phi = [&](const double alpha) { return f.At(x + alpha * d); };
  const auto n = f.Size();
  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::debug("Iteration {}, f(x)={}", k, f.At(x));
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
      d += w_prev * prev_d;
      if (d.dot(grad) >= 0) d = -grad;
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
  if (f.Size() == 0 || x0.size() != static_cast<Eigen::Index>(f.Size()) ||
      !x0.allFinite()) {
    throw std::invalid_argument("Invalid initial point");
  }
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd grad = f.Gradient(x), grad_next;
  Eigen::VectorXd d;

  const auto phi = [&](const double alpha) { return f.At(x + alpha * d); };

  const auto n = f.Size();
  Eigen::MatrixXd g(n, n);
  g.setIdentity();

  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::debug("Iteration {}, f(x)={}", k, f.At(x));
    if (grad.norm() < grad_epsilon) {
      spdlog::debug("||grad|| < epsilon");
      return x;
    }

    d = -g * grad;
    if (!d.allFinite() || d.dot(grad) >= 0) {
      g.setIdentity();
      d = -grad;
    }
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
    const double curvature1 = w1.dot(delta_grad);
    const double curvature2 = w2.dot(delta_grad);
    if (curvature1 > 1e-12 * w1.norm() * delta_grad.norm() &&
        curvature2 > 1e-12 * w2.norm() * delta_grad.norm()) {
      g += w1 * w1.transpose() / curvature1 - w2 * w2.transpose() / curvature2;
    } else {
      g.setIdentity();
    }

    x = std::move(x_next);
    grad = std::move(grad_next);
  }

  return x;
}

Eigen::VectorXd LevenbergMarquardt(const IMultivariateFunction& f,
                                   const Eigen::VectorXd& x0,
                                   const std::size_t max_iterations,
                                   const double epsilon) {
  if (f.Size() == 0 || x0.size() != static_cast<Eigen::Index>(f.Size()) ||
      !x0.allFinite()) {
    throw std::invalid_argument("Invalid initial point");
  }
  auto x = x0;
  auto mu = 1e+4;
  const auto n = f.Size();

  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::debug("Iteration {}, f(x)={}", k, f.At(x));
    const auto grad = f.Gradient(x);
    if (grad.norm() < epsilon) {
      spdlog::debug("||grad|| < epsilon");
      return x;
    }

    const auto f_x = f.At(x);
    const Eigen::MatrixXd damped =
        f.Hessian(x) + mu * Eigen::MatrixXd::Identity(n, n);
    const Eigen::VectorXd candidate = x - damped.ldlt().solve(grad);
    if (candidate.allFinite() && f.At(candidate) < f_x) {
      x = candidate;
      mu /= 2;
    } else {
      mu *= 2;
      if (!std::isfinite(mu)) throw std::runtime_error("Damping overflow");
    }
  }

  return x;
}

}  // namespace hw3
