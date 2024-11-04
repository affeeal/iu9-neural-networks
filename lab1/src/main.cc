#include <memory>

// clang-format off
#include <matplot/matplot.h>
// clang-format on

#include "activation_function.h"
#include "cost_function.h"
#include "data_supplier.h"
#include "perceptron.h"

namespace {

constexpr std::size_t kEpochs = 200;
constexpr static auto kCfg = nn::Config{
    .epochs = kEpochs,
    .mini_batch_size = 10,
    .eta = 0.5,
    .monitor_training_cost = true,
    .monitor_training_accuracy = true,
    .monitor_testing_cost = true,
    .monitor_testing_accuracy = true,
};

void RunSigmoidMSE() {
  const auto data_supplier = nn::DataSupplier(0.0, 1.0);
  const auto training = data_supplier.GetTrainingData();
  const auto testing = data_supplier.GetTestingData();

  auto cost_function = std::make_unique<nn::MSE>();
  auto activation_functions =
      std::vector<std::unique_ptr<nn::IActivationFunction>>{};
  activation_functions.push_back(std::make_unique<nn::Sigmoid>());
  const auto layers_sizes =
      std::vector<std::size_t>{nn::kScanSize, nn::kClassesCount};

  auto perceptron = nn::Perceptron(
      std::move(cost_function), std::move(activation_functions), layers_sizes);
  const auto metrics =
      perceptron.StochasticGradientSearch(training, testing, kCfg);

  const auto x = matplot::linspace(0, kEpochs);
  matplot::plot(x, metrics.training_cost, x, metrics.testing_cost);
  matplot::title("Sigmoid + MSE training, testing cost");
  matplot::show();

  matplot::plot(x, metrics.training_accuracy, x, metrics.testing_accuracy);
  matplot::title("Sigmoid + MSE training, testing accuracy");
  matplot::show();
}

void RunTanhMSE() {
  const auto data_supplier = nn::DataSupplier(-1.0, 1.0);
  const auto training = data_supplier.GetTrainingData();
  const auto testing = data_supplier.GetTestingData();

  auto cost_function = std::make_unique<nn::MSE>();
  auto activation_functions =
      std::vector<std::unique_ptr<nn::IActivationFunction>>{};
  activation_functions.push_back(std::make_unique<nn::Tanh>());
  const auto layers_sizes =
      std::vector<std::size_t>{nn::kScanSize, nn::kClassesCount};

  auto perceptron = nn::Perceptron(
      std::move(cost_function), std::move(activation_functions), layers_sizes);
  const auto metrics =
      perceptron.StochasticGradientSearch(training, testing, kCfg);

  const auto x = matplot::linspace(0, kEpochs);
  matplot::plot(x, metrics.training_cost, x, metrics.testing_cost);
  matplot::title("tanh + MSE training, testing cost");
  matplot::show();

  matplot::plot(x, metrics.training_accuracy, x, metrics.testing_accuracy);
  matplot::title("tanh + MSE training, testing accuracy");
  matplot::show();
}

void RunSoftmaxCrossEntropy() {
  const auto data_supplier = nn::DataSupplier(0.0, 1.0);
  const auto training = data_supplier.GetTrainingData();
  const auto testing = data_supplier.GetTestingData();

  auto cost_function = std::make_unique<nn::CrossEntropy>();
  auto activation_functions =
      std::vector<std::unique_ptr<nn::IActivationFunction>>{};
  activation_functions.push_back(std::make_unique<nn::Softmax>());
  const auto layers_sizes =
      std::vector<std::size_t>{nn::kScanSize, nn::kClassesCount};

  auto perceptron = nn::Perceptron(
      std::move(cost_function), std::move(activation_functions), layers_sizes);
  const auto metrics =
      perceptron.StochasticGradientSearch(training, testing, kCfg);

  const auto x = matplot::linspace(0, kEpochs);
  matplot::plot(x, metrics.training_cost, x, metrics.testing_cost);
  matplot::title("Softmax + Cross-entropy training, testing cost");
  matplot::show();

  matplot::plot(x, metrics.training_accuracy, x, metrics.testing_accuracy);
  matplot::title("Softmax + Cross-entropy training, testing accuracy");
  matplot::show();
}

}  // namespace

int main(int argc, char* argv[]) { RunSoftmaxCrossEntropy(); }
