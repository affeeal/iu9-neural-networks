#ifdef NN_ENABLE_PLOTS
#include <matplot/matplot.h>
#endif
#include <iostream>
#include <memory>
#include <string_view>

#include "activation_function.h"
#include "cost_function.h"
#include "data_supplier.h"
#include "perceptron.h"

namespace {

std::string test_path;
std::string train_path;

constexpr std::size_t kHiddenLayerSize = 40;
constexpr static auto kCfg = nn::Config{
    .epochs = 200,
    .mini_batch_size = 100,
    .eta = 0.025,
    .monitor_training_cost = true,
    .monitor_training_accuracy = true,
    .monitor_testing_cost = true,
    .monitor_testing_accuracy = true,
};

void RunLeakyReluSoftmaxMSE() {
  const auto data_supplier = hw2::DataSupplier(train_path, test_path, 0.0, 1.0);
  const auto training = data_supplier.GetTrainingData();
  const auto testing = data_supplier.GetTestingData();

  auto cost_function = std::make_unique<nn::MSE>();
  auto activation_functions =
      std::vector<std::unique_ptr<nn::IActivationFunction>>{};
  activation_functions.push_back(std::make_unique<nn::LeakyReLU>(0.01));
  activation_functions.push_back(std::make_unique<nn::LeakyReLU>(0.01));
  activation_functions.push_back(std::make_unique<nn::LeakyReLU>(0.01));
  activation_functions.push_back(std::make_unique<nn::Softmax>());
  const auto layers_sizes = std::vector<std::size_t>{
      hw2::kScanSize, kHiddenLayerSize, kHiddenLayerSize, kHiddenLayerSize,
      hw2::kDigitsNumber};

  auto perceptron = nn::Perceptron(
      std::move(cost_function), std::move(activation_functions), layers_sizes);
  const auto metrics =
      perceptron.StochasticGradientSearch(training, testing, kCfg);

#ifdef NN_ENABLE_PLOTS
  const auto x = matplot::linspace(1, kCfg.epochs, kCfg.epochs);
  matplot::plot(x, metrics.training_cost, x, metrics.testing_cost);
  matplot::title("Leaky ReLU, Softmax + MSE training, testing cost");
  matplot::show();

  matplot::plot(x, metrics.training_accuracy, x, metrics.testing_accuracy);
  matplot::title("Leaky ReLU, Softmax + MSE training, testing accuracy");
  matplot::show();
#endif
}

void RunLeakyReluSoftmaxCrossEntropy() {
  const auto data_supplier = hw2::DataSupplier(train_path, test_path, 0.0, 1.0);
  const auto training = data_supplier.GetTrainingData();
  const auto testing = data_supplier.GetTestingData();

  auto cost_function = std::make_unique<nn::CrossEntropy>();
  auto activation_functions =
      std::vector<std::unique_ptr<nn::IActivationFunction>>{};
  activation_functions.push_back(std::make_unique<nn::LeakyReLU>(0.01));
  activation_functions.push_back(std::make_unique<nn::LeakyReLU>(0.01));
  activation_functions.push_back(std::make_unique<nn::LeakyReLU>(0.01));
  activation_functions.push_back(std::make_unique<nn::Softmax>());
  const auto layers_sizes = std::vector<std::size_t>{
      hw2::kScanSize, kHiddenLayerSize, kHiddenLayerSize, kHiddenLayerSize,
      hw2::kDigitsNumber};

  auto perceptron = nn::Perceptron(
      std::move(cost_function), std::move(activation_functions), layers_sizes);
  const auto metrics =
      perceptron.StochasticGradientSearch(training, testing, kCfg);

#ifdef NN_ENABLE_PLOTS
  const auto x = matplot::linspace(1, kCfg.epochs, kCfg.epochs);
  matplot::plot(x, metrics.training_cost, x, metrics.testing_cost);
  matplot::title("Leaky ReLU, Softmax + Cross-entropy training, testing cost");
  matplot::show();

  matplot::plot(x, metrics.training_accuracy, x, metrics.testing_accuracy);
  matplot::title(
      "Leaky ReLU, Softmax + Cross-entropy training, testing accuracy");
  matplot::show();
#endif
}

void RunLeakyReluSoftmaxKlDivergence() {
  const auto data_supplier = hw2::DataSupplier(train_path, test_path, 0.0, 1.0);
  const auto training = data_supplier.GetTrainingData();
  const auto testing = data_supplier.GetTestingData();

  auto cost_function = std::make_unique<nn::KLDivergence>();
  auto activation_functions =
      std::vector<std::unique_ptr<nn::IActivationFunction>>{};
  activation_functions.push_back(std::make_unique<nn::LeakyReLU>(0.01));
  activation_functions.push_back(std::make_unique<nn::LeakyReLU>(0.01));
  activation_functions.push_back(std::make_unique<nn::LeakyReLU>(0.01));
  activation_functions.push_back(std::make_unique<nn::Softmax>());
  const auto layers_sizes = std::vector<std::size_t>{
      hw2::kScanSize, kHiddenLayerSize, kHiddenLayerSize, kHiddenLayerSize,
      hw2::kDigitsNumber};

  auto perceptron = nn::Perceptron(
      std::move(cost_function), std::move(activation_functions), layers_sizes);
  const auto metrics =
      perceptron.StochasticGradientSearch(training, testing, kCfg);

#ifdef NN_ENABLE_PLOTS
  const auto x = matplot::linspace(1, kCfg.epochs, kCfg.epochs);
  matplot::plot(x, metrics.training_cost, x, metrics.testing_cost);
  matplot::title(
      "Leaky ReLU, Softmax + K.-L. Divergence training, testing cost");
  matplot::show();

  matplot::plot(x, metrics.training_accuracy, x, metrics.testing_accuracy);
  matplot::title(
      "Leaky ReLU, Softmax + K.-L. Divergence training, testing accuracy");
  matplot::show();
#endif
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout << "Usage: hw2 TRAIN.csv TEST.csv\n";
    return 0;
  }
  if (argc != 3) {
    std::cerr << "Usage: hw2 TRAIN.csv TEST.csv\n";
    return 1;
  }
  train_path = argv[1];
  test_path = argv[2];
  try {
    RunLeakyReluSoftmaxMSE();
    RunLeakyReluSoftmaxCrossEntropy();
    RunLeakyReluSoftmaxKlDivergence();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
