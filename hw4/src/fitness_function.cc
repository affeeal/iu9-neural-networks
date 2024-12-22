#include "fitness_function.h"

#include <cassert>

#include "chromosome.h"
#include "cost_function.h"
#include "perceptron.h"

namespace nn {

namespace {

double CostToFitness(const double value) {
  return std::exp(1 / (value + 1e-8));
}

}  // namespace

ISgdFitness::ISgdFitness(std::unique_ptr<IDataSupplier>&& data_supplier)
    : data_supplier_(std::move(data_supplier)) {}

double SgdFitness::Assess(const IChromosome& chromosome) const {
  const auto kit = static_cast<const SgdHyperparametersKit&>(chromosome);

  auto cost_function = std::make_unique<CrossEntropy>();

  auto activation_functions =
      std::vector<std::unique_ptr<IActivationFunction>>{};
  const auto hidden_layers = kit.get_hidden_layers();
  activation_functions.reserve(hidden_layers + 1);
  for (std::size_t i = 0; i < hidden_layers; ++i) {
    activation_functions.push_back(std::make_unique<LeakyReLU>(0.01));
  }
  activation_functions.push_back(std::make_unique<Softmax>());

  auto layers_sizes = std::vector<std::size_t>{};
  layers_sizes.reserve(hidden_layers + 2);
  layers_sizes.push_back(data_supplier_->GetInputLayerSize());
  const auto neurons_per_hidden_layer = kit.get_neurons_per_hidden_layer();
  for (std::size_t i = 0; i < hidden_layers; ++i) {
    layers_sizes.push_back(neurons_per_hidden_layer);
  }
  layers_sizes.push_back(data_supplier_->GetOutputLayerSize());

  auto perceptron = Perceptron(std::move(cost_function),
                               std::move(activation_functions), layers_sizes);
  const auto cfg = SgdConfiguration{
      .epochs = kit.get_epochs(),
      .mini_batch_size = kit.get_mini_batch_size(),
      .learning_rate = kit.get_learning_rate(),
      .monitor_train_cost = true,
      .monitor_train_accuracy = true,
      .monitor_test_cost = true,
      .monitor_test_accuracy = true,
  };

  const auto train_data = data_supplier_->GetTrainData();
  const auto test_data = data_supplier_->GetTestData();

  auto metrics = perceptron.Sgd(train_data, test_data, cfg);
  return CostToFitness(metrics.test_cost.back());
}

double SgdNagFitness::Assess(const IChromosome& chromosome) const {
  constexpr double kGamma = 0.9;

  const auto kit = static_cast<const SgdHyperparametersKit&>(chromosome);

  auto cost_function = std::make_unique<CrossEntropy>();

  auto activation_functions =
      std::vector<std::unique_ptr<IActivationFunction>>{};
  const auto hidden_layers = kit.get_hidden_layers();
  activation_functions.reserve(hidden_layers + 1);
  for (std::size_t i = 0; i < hidden_layers; ++i) {
    activation_functions.push_back(std::make_unique<LeakyReLU>(0.01));
  }
  activation_functions.push_back(std::make_unique<Softmax>());

  auto layers_sizes = std::vector<std::size_t>{};
  layers_sizes.reserve(hidden_layers + 2);
  layers_sizes.push_back(data_supplier_->GetInputLayerSize());
  const auto neurons_per_hidden_layer = kit.get_neurons_per_hidden_layer();
  for (std::size_t i = 0; i < hidden_layers; ++i) {
    layers_sizes.push_back(neurons_per_hidden_layer);
  }
  layers_sizes.push_back(data_supplier_->GetOutputLayerSize());

  auto perceptron = Perceptron(std::move(cost_function),
                               std::move(activation_functions), layers_sizes);
  const auto cfg = SgdConfiguration{
      .epochs = kit.get_epochs(),
      .mini_batch_size = kit.get_mini_batch_size(),
      .learning_rate = kit.get_learning_rate(),
      .monitor_train_cost = true,
      .monitor_train_accuracy = true,
      .monitor_test_cost = true,
      .monitor_test_accuracy = true,
  };

  const auto train_data = data_supplier_->GetTrainData();
  const auto test_data = data_supplier_->GetTestData();

  auto metrics = perceptron.SgdNag(train_data, test_data, cfg, kGamma);
  return CostToFitness(metrics.test_cost.back());
}

double SgdAdagradFitness::Assess(const IChromosome& chromosome) const {
  constexpr double kEpsilon = 1e-8;

  const auto kit = static_cast<const SgdHyperparametersKit&>(chromosome);

  auto cost_function = std::make_unique<CrossEntropy>();

  auto activation_functions =
      std::vector<std::unique_ptr<IActivationFunction>>{};
  const auto hidden_layers = kit.get_hidden_layers();
  activation_functions.reserve(hidden_layers + 1);
  for (std::size_t i = 0; i < hidden_layers; ++i) {
    activation_functions.push_back(std::make_unique<LeakyReLU>(0.01));
  }
  activation_functions.push_back(std::make_unique<Softmax>());

  auto layers_sizes = std::vector<std::size_t>{};
  layers_sizes.reserve(hidden_layers + 2);
  layers_sizes.push_back(data_supplier_->GetInputLayerSize());
  const auto neurons_per_hidden_layer = kit.get_neurons_per_hidden_layer();
  for (std::size_t i = 0; i < hidden_layers; ++i) {
    layers_sizes.push_back(neurons_per_hidden_layer);
  }
  layers_sizes.push_back(data_supplier_->GetOutputLayerSize());

  auto perceptron = Perceptron(std::move(cost_function),
                               std::move(activation_functions), layers_sizes);
  const auto cfg = SgdConfiguration{
      .epochs = kit.get_epochs(),
      .mini_batch_size = kit.get_mini_batch_size(),
      .learning_rate = kit.get_learning_rate(),
      .monitor_train_cost = true,
      .monitor_train_accuracy = true,
      .monitor_test_cost = true,
      .monitor_test_accuracy = true,
  };

  const auto train_data = data_supplier_->GetTrainData();
  const auto test_data = data_supplier_->GetTestData();

  auto metrics = perceptron.SgdAdagrad(train_data, test_data, cfg, kEpsilon);
  return CostToFitness(metrics.test_cost.back());
}

double SgdAdamFitness::Assess(const IChromosome& chromosome) const {
  constexpr double kEpsilon = 1e-8;
  constexpr double kBeta1 = 0.9;
  constexpr double kBeta2 = 0.999;

  const auto kit = static_cast<const SgdHyperparametersKit&>(chromosome);

  auto cost_function = std::make_unique<CrossEntropy>();

  auto activation_functions =
      std::vector<std::unique_ptr<IActivationFunction>>{};
  const auto hidden_layers = kit.get_hidden_layers();
  activation_functions.reserve(hidden_layers + 1);
  for (std::size_t i = 0; i < hidden_layers; ++i) {
    activation_functions.push_back(std::make_unique<LeakyReLU>(0.01));
  }
  activation_functions.push_back(std::make_unique<Softmax>());

  auto layers_sizes = std::vector<std::size_t>{};
  layers_sizes.reserve(hidden_layers + 2);
  layers_sizes.push_back(data_supplier_->GetInputLayerSize());
  const auto neurons_per_hidden_layer = kit.get_neurons_per_hidden_layer();
  for (std::size_t i = 0; i < hidden_layers; ++i) {
    layers_sizes.push_back(neurons_per_hidden_layer);
  }
  layers_sizes.push_back(data_supplier_->GetOutputLayerSize());

  auto perceptron = Perceptron(std::move(cost_function),
                               std::move(activation_functions), layers_sizes);
  const auto cfg = SgdConfiguration{
      .epochs = kit.get_epochs(),
      .mini_batch_size = kit.get_mini_batch_size(),
      .learning_rate = kit.get_learning_rate(),
      .monitor_train_cost = true,
      .monitor_train_accuracy = true,
      .monitor_test_cost = true,
      .monitor_test_accuracy = true,
  };

  const auto train_data = data_supplier_->GetTrainData();
  const auto test_data = data_supplier_->GetTestData();

  auto metrics =
      perceptron.SgdAdam(train_data, test_data, cfg, kBeta1, kBeta2, kEpsilon);
  return CostToFitness(metrics.test_cost.back());
}

}  // namespace nn
