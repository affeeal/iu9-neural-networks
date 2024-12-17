#include "fitness_function.h"

#include <cassert>

#include "chromosome.h"
#include "cost_function.h"
#include "perceptron.h"

namespace nn {

AccuracyOnTestData::AccuracyOnTestData(
    std::unique_ptr<IDataSupplier>&& data_supplier)
    : data_supplier_(std::move(data_supplier)) {}

double AccuracyOnTestData::Assess(const IChromosome& chromosome) const {
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
  return std::exp(1 / (metrics.test_cost.back() + 1e-8));
}

}  // namespace nn
