#include <memory>

#include "activation_function.h"
#include "cost_function.h"
#include "data_supplier.h"
#include "perceptron.h"

namespace {

constexpr std::size_t kEpochs = 100;
constexpr double kEta = 0.01;
constexpr std::size_t kMiniBatchSize = 5;

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
  const auto layers_sizes = std::vector<std::size_t>{20, 10};

  auto perceptron = lab1::Perceptron(
      std::move(cost_function), std::move(activation_functions), layers_sizes);
  perceptron.StochasticGradientSearch(training, kEpochs, kMiniBatchSize, kEta,
                                      testing);
}
