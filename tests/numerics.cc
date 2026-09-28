#include <memory>
#include <vector>

#include "check.h"
#include "output_layer.h"

int main() {
  std::vector<std::unique_ptr<nn::IActivationFunction>> functions;
  functions.push_back(std::make_unique<nn::Linear>());
  functions.push_back(std::make_unique<nn::ReLU>());
  functions.push_back(std::make_unique<nn::LeakyReLU>(0.1));
  functions.push_back(std::make_unique<nn::Sigmoid>());
  functions.push_back(std::make_unique<nn::Tanh>());
  functions.push_back(std::make_unique<nn::Softmax>());
  const Eigen::Vector3d z(-1.3, 0.2, 2.0);
  constexpr double h = 1e-5;
  for (auto& f : functions) {
    const auto jacobian = f->Jacobian(z);
    CHECK(jacobian.rows() == 3 && jacobian.cols() == 3);
    for (Eigen::Index j = 0; j < 3; ++j) {
      Eigen::VectorXd plus = z, minus = z;
      plus[j] += h;
      minus[j] -= h;
      const Eigen::VectorXd numerical =
          (f->Apply(plus) - f->Apply(minus)) / (2 * h);
      Near((jacobian.col(j) - numerical).norm(), 0, 1e-9);
    }
    CHECK(f->Apply(Eigen::Vector3d(-1000, 0, 1000)).allFinite());
  }

  nn::Softmax softmax;
  nn::CrossEntropy ce;
  nn::KLDivergence kl;
  nn::MSE mse;
  const Eigen::Vector3d y(0, 1, 0), a(0.2, 0.3, 0.5);
  for (nn::ICostFunction* loss :
       std::vector<nn::ICostFunction*>{&ce, &kl, &mse}) {
    const auto gradient = loss->GradientWrtActivations(y, a);
    for (Eigen::Index j = 0; j < 3; ++j) {
      Eigen::VectorXd plus = a, minus = a;
      plus[j] += h;
      minus[j] -= h;
      Near(gradient[j],
           (loss->Apply(y, plus) - loss->Apply(y, minus)) / (2 * h), 1e-8);
    }
  }
  Near(ce.Apply(y, a), -std::log(0.3));
  Near(kl.Apply(y, a), ce.Apply(y, a));
  Near(kl.Apply(a, a), 0);
  Near(ce.Apply(y, y), 0);
  Near(kl.Apply(y, y), 0);
  CHECK(ce.GradientWrtActivations(y, y).allFinite());
  // Wrong confident prediction: loss and fused gradient remain finite.
  const Eigen::Vector3d extreme(1000, -1000, 0);
  for (nn::ICostFunction* loss : std::vector<nn::ICostFunction*>{&ce, &kl}) {
    Near(nn::OutputCost(softmax, *loss, y, extreme), 2000);
    const auto delta =
        nn::OutputDelta(softmax, *loss, y, extreme, softmax.Apply(extreme));
    Near((delta - Eigen::Vector3d(1, -1, 0)).norm(), 0);
    const Eigen::Vector3d soft_targets(0.2, 0.3, 0.5);
    const auto d =
        nn::OutputDelta(softmax, *loss, soft_targets, z, softmax.Apply(z));
    for (Eigen::Index j = 0; j < 3; ++j) {
      Eigen::VectorXd plus = z, minus = z;
      plus[j] += h;
      minus[j] -= h;
      Near(d[j],
           (nn::OutputCost(softmax, *loss, soft_targets, plus) -
            nn::OutputCost(softmax, *loss, soft_targets, minus)) /
               (2 * h),
           1e-8);
    }
  }
  Near((softmax.Apply(z) - softmax.Apply((z.array() + 1000).matrix())).norm(),
       0, 1e-12);
  Throws([&] { softmax.Apply(Eigen::VectorXd{}); });
}
