#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <iostream>

namespace {

class IFunction {
 public:
  virtual ~IFunction() = default;

 public:
  virtual std::size_t Size() const = 0;

  virtual double Calculate(const Eigen::VectorXd& u) const = 0;

  virtual Eigen::VectorXd Gradient(const Eigen::VectorXd& u) const = 0;
};

class RosenbrockFunction final : public IFunction {
 public:
  static constexpr std::size_t kInputSize = 2;

  std::size_t Size() const override { return kInputSize; }

  double Calculate(const Eigen::VectorXd& u) const override {
    assert(u.size() == Size());
    return 250 * std::pow(std::pow(u.x(), 2) - u.y(), 2) +
           2 * std::pow(u.x() - 1, 2) + 50;
  }

  Eigen::VectorXd Gradient(const Eigen::VectorXd& u) const override {
    assert(u.size() == Size());
    return Eigen::Vector<double, kInputSize>{
        1000 * std::pow(u.x(), 3) - 1000 * u.x() * u.y() + 4 * u.x() - 4,
        -500 * std::pow(u.x(), 2) + 500 * u.y()};
  }
};

class Paraboloid final : public IFunction {
 public:
  static constexpr std::size_t kInputSize = 2;

  std::size_t Size() const override { return kInputSize; }

  double Calculate(const Eigen::VectorXd& u) const override {
    assert(u.size() == Size());
    return std::pow(u.x() - 1, 2) + std::pow(u.y() - 1, 2);
  }

  Eigen::VectorXd Gradient(const Eigen::VectorXd& u) const override {
    assert(u.size() == Size());
    return Eigen::Vector<double, kInputSize>{2 * (u.x() - 1), 2 * (u.y() - 1)};
  }
};

struct Config final {
  double a, b;
  double delta, epsilon;
};

double DichotomyMinimum(const std::function<double(double)>& f,
                        const Config& cfg) {
  spdlog::debug("Dichotomy minimum...");
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
  spdlog::info("Fletcher-Reeves");
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd prev_grad, grad;
  Eigen::VectorXd prev_d, d;
  double alpha;
  const auto phi = [&](const double alpha) {
    return f.Calculate(x + alpha * d);
  };
  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::info("Iteration {}, x = ({}, {}), f(x) = {}", k, x.x(), x.y(),
                 f.Calculate(x));
    grad = f.Gradient(x);
    if (grad.norm() < grad_epsilon) {
      spdlog::info("||grad|| < epsilon");
      break;
    }

    d = -grad;
    if (k > 0) {
      const auto w_prev = grad.squaredNorm() / prev_grad.squaredNorm();
      d += w_prev * prev_grad;
    }

    alpha = DichotomyMinimum(phi, cfg);
    x_next = x + alpha * d;

    if ((x_next - x).norm() < delta &&
        std::abs(f.Calculate(x_next) - f.Calculate(x)) < epsilon) {
      spdlog::info("|x - x_next| < delta, |f(x) - f(x_next)| < epsilon");
      break;
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
  spdlog::info("Polak-Ribier");
  Eigen::VectorXd x = x0, x_next;
  Eigen::VectorXd prev_grad, grad;
  Eigen::VectorXd prev_d, d;
  double alpha;
  const auto phi = [&](const double alpha) {
    return f.Calculate(x + alpha * d);
  };
  const auto n = f.Size();
  for (std::size_t k = 0; k < max_iterations; ++k) {
    spdlog::info("Iteration {}, x = ({}, {}), f(x) = {}", k, x.x(), x.y(),
                 f.Calculate(x));
    grad = f.Gradient(x);
    if (grad.norm() < grad_epsilon) {
      spdlog::info("||grad|| < epsilon");
      break;
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
      spdlog::info("|x - x_next| < delta, |f(x) - f(x_next)| < epsilon");
      break;
    }

    x = std::move(x_next);
    prev_grad = std::move(grad);
    prev_d = std::move(d);
  }

  return x;
}

}  // namespace

int main(int argc, char* argv[]) {
  spdlog::set_level(spdlog::level::level_enum::debug);
  const auto f = Paraboloid();
  const auto x0 = Eigen::Vector<double, 2>{100, 100};
  const auto cfg =
      Config{.a = -10.0, .b = 10.0, .delta = 10e-16, .epsilon = 10e-15};
  std::cout << FletcherReeves(f, x0, 100, 10e-15, 10e-15, 10e-15, cfg) << '\n';
}
