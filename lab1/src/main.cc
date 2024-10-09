#include <matplot/matplot.h>

#include <memory>

#include "activation_function.h"
#include "cost_function.h"
#include "data_supplier.h"
#include "perceptron.h"

namespace {

constexpr std::size_t kEpochs = 200;
constexpr double kEta = 0.5;
constexpr std::size_t kMiniBatchSize = 1;
constexpr auto kParam = lab1::Parametrization{
    .epochs = kEpochs,
    .mini_batch_size = kMiniBatchSize,
    .eta = kEta,
    .monitor_training_cost = true,
    .monitor_training_accuracy = true,
    .monitor_testing_cost = true,
    .monitor_testing_accuracy = true,
};

}  // namespace

int main(int argc, char* argv[]) {
  const auto data_supplier = lab1::DataSupplier(0.0, 1.0);

  const auto training = data_supplier.GetTrainingData();
  const auto validation = data_supplier.GetValidationData();
  const auto testing = data_supplier.GetTestingData();

  auto cost_function = std::make_unique<lab1::MSE>();
  auto activation_functions =
      std::vector<std::unique_ptr<lab1::IActivationFunction>>{};
  activation_functions.push_back(std::make_unique<lab1::Sigmoid>());
  const auto layers_sizes = std::vector<std::size_t>{lab1::kScanSize, 20};

  auto perceptron = lab1::Perceptron(
      std::move(cost_function), std::move(activation_functions), layers_sizes);
  const auto metrics =
      perceptron.StochasticGradientSearch(training, testing, kParam);

  const auto x = matplot::linspace(0, kEpochs);
  matplot::plot(x, metrics.training_cost, x, metrics.testing_cost);
  matplot::title("Training, testing cost");
  matplot::show();

  matplot::plot(x, metrics.training_accuracy, x, metrics.testing_accuracy);
  matplot::title("Training, testing accuracy");
  matplot::show();
}
