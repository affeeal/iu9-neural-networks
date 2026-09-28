#include <spdlog/spdlog.h>

#include "optimization.h"

int main() {
  using namespace hw3;
  spdlog::set_level(spdlog::level::info);

  const auto f = RosenbrockFunction();
  const auto x0 = Eigen::Vector<double, 2>{100, 100};

  constexpr std::size_t kMaxIterations = 100;
  constexpr auto kDelta = 1e-10;
  constexpr auto kEpsilon = 1e-9;
  constexpr auto kGradientEpsilon = 1e-9;
  constexpr auto kA = 0.0;
  constexpr auto kB = 10.0;
  spdlog::info("Reference minimum: f(1, 1) = 50; iteration budget: {}",
               kMaxIterations);

  auto u = GradientDescent(f, x0, kMaxIterations, kGradientEpsilon, kDelta,
                           kEpsilon, kA, kB);
  spdlog::info("Gradient descent: x=({}, {}), f(x)={}", u.x(), u.y(), f.At(u));

  u = FletcherReeves(f, x0, kMaxIterations, kGradientEpsilon, kDelta, kEpsilon,
                     kA, kB);
  spdlog::info("Fletcher-Reeves: x=({}, {}), f(x)={}", u.x(), u.y(), f.At(u));

  u = PolakRibier(f, x0, kMaxIterations, kGradientEpsilon, kDelta, kEpsilon, kA,
                  kB);
  spdlog::info("Polak-Ribier: x=({}, {}), f(x)={}", u.x(), u.y(), f.At(u));

  u = DavidonFletcherPowell(f, x0, kMaxIterations, kGradientEpsilon, kDelta,
                            kEpsilon, kA, kB);
  spdlog::info("Davidon-Fletcher-Powell: x=({}, {}), f(x)={}", u.x(), u.y(),
               f.At(u));

  u = LevenbergMarquardt(f, x0, kMaxIterations, kGradientEpsilon);
  spdlog::info("Levenberg-Marquardt: x=({}, {}), f(x)={}", u.x(), u.y(),
               f.At(u));
}
