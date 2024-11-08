#include <matplot/matplot.h>

#include <memory>

#include "activation_function.h"
#include "cost_function.h"
#include "data_supplier.h"
#include "perceptron.h"

namespace {

const std::string kDefaultTestPath = "../data/mnist_test.csv";
const std::string kDefaultTrainPath = "../data/mnist_train.csv";

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

void Run3LeakyReluSoftmaxMSE() {
  const auto data_supplier =
      hw2::DataSupplier(kDefaultTrainPath, kDefaultTestPath, 0.0, 1.0);
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

  const auto x = matplot::linspace(0, kCfg.epochs);
  matplot::plot(x, metrics.training_cost, x, metrics.testing_cost);
  matplot::title("Leaky ReLU, Softmax + MSE training, testing cost");
  matplot::show();

  matplot::plot(x, metrics.training_accuracy, x, metrics.testing_accuracy);
  matplot::title("Leaky ReLU, Softmax + MSE training, testing accuracy");
  matplot::show();
}

void Run3LeakyReluSoftmaxCrossEntropy() {
  const auto data_supplier =
      hw2::DataSupplier(kDefaultTrainPath, kDefaultTestPath, 0.0, 1.0);
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

  const auto x = matplot::linspace(0, kCfg.epochs);
  matplot::plot(x, metrics.training_cost, x, metrics.testing_cost);
  matplot::title("Leaky ReLU, Softmax + Cross-entropy training, testing cost");
  matplot::show();

  matplot::plot(x, metrics.training_accuracy, x, metrics.testing_accuracy);
  matplot::title(
      "Leaky ReLU, Softmax + Cross-entropy training, testing accuracy");
  matplot::show();
}

void Run3LeakyReluSoftmaxKlDivergence() {
  const auto data_supplier =
      hw2::DataSupplier(kDefaultTrainPath, kDefaultTestPath, 10e-6, 1.0);
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

  const auto x = matplot::linspace(0, kCfg.epochs);
  matplot::plot(x, metrics.training_cost, x, metrics.testing_cost);
  matplot::title(
      "Leaky ReLU, Softmax + K.-L. Divergence training, testing cost");
  matplot::show();

  matplot::plot(x, metrics.training_accuracy, x, metrics.testing_accuracy);
  matplot::title(
      "Leaky ReLU, Softmax + K.-L. Divergence training, testing accuracy");
  matplot::show();
}

}  // namespace

int main(int argc, char *argv[]) {
  Run3LeakyReluSoftmaxMSE();
  Run3LeakyReluSoftmaxCrossEntropy();
  Run3LeakyReluSoftmaxKlDivergence();
}

/*
 * 1x20
 * Epoch 199; training cost: 0.166286; training accuracy: 47518/50000; testing
 * cost: 0.193274; testing accuracy: 9421/10000; Epoch 199; training cost:
 * 0.192402; training accuracy: 47194/50000; testing cost: 0.217352; testing
 * accuracy: 9345/10000; Epoch 199; training cost: 0.17601; training accuracy:
 * 47346/50000; testing cost: 0.201559; testing accuracy: 9415/10000;
 *
 * 1x40
 * Epoch 199; training cost: 0.123245; training accuracy: 48121/50000; testing
 * cost: 0.161724; testing accuracy: 9527/10000; Epoch 199; training cost:
 * 0.131022; training accuracy: 48097/50000; testing cost: 0.173189; testing
 * accuracy: 9483/10000; Epoch 199; training cost: 0.118821; training accuracy:
 * 48233/50000; testing cost: 0.168346; testing accuracy: 9506/10000;
 *
 * 3x20
 * Epoch 199; training cost: 0.163502; training accuracy: 47504/50000; testing
 * cost: 0.206251; testing accuracy: 9416/10000; Epoch 199; training cost:
 * 0.157613; training accuracy: 47681/50000; testing cost: 0.216353; testing
 * accuracy: 9386/10000; Epoch 199; training cost: 0.163792; training accuracy:
 * 47487/50000; testing cost: 0.213412; testing accuracy: 9369/10000;
 *
 * 3x40
 * Epoch 199; training cost: 0.105351; training accuracy: 48422/50000; testing
 * cost: 0.190718; testing accuracy: 9481/10000; Epoch 199; training cost:
 * 0.118457; training accuracy: 48232/50000; testing cost: 0.17582; testing
 * accuracy: 9502/10000; Epoch 199; training cost: 0.111865; training accuracy:
 * 48315/50000; testing cost: 0.197848; testing accuracy: 9457/10000;
 */
