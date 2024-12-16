#include "fitness_function.h"

#include <stdexcept>

#include "cost_function.h"
#include "perceptron.h"

namespace nn {

Chromosome::Chromosome(std::vector<double>&& genes)
    : genes_(std::move(genes)) {}

const std::vector<double>& Chromosome::get_genes() const { return genes_; }

AccuracyOnTestData::AccuracyOnTestData(
    std::unique_ptr<IDataSupplier>&& data_supplier)
    : data_supplier_(std::move(data_supplier)) {}

double AccuracyOnTestData::Assess(const Chromosome& chromosome) const {
  // TODO: Provide an adequate chromosome interpretation.
  const auto& hyperparameters = chromosome.get_genes();
  if (hyperparameters.size() != kHyperparametersNumber) {
    throw std::runtime_error(
        "Expected " + std::to_string(kHyperparametersNumber) +
        " hyperparameters, got " + std::to_string(hyperparameters.size()));
  }

  auto cost_function = std::make_unique<CrossEntropy>();

  auto activation_functions =
      std::vector<std::unique_ptr<IActivationFunction>>{};
  const auto hidden_layers =
      static_cast<std::size_t>(hyperparameters.at(Index::kHiddenLayers));
  activation_functions.reserve(hidden_layers + 1);
  for (std::size_t i = 0; i < hidden_layers; ++i) {
    activation_functions.push_back(std::make_unique<LeakyReLU>(0.01));
  }
  activation_functions.push_back(std::make_unique<Softmax>());

  auto layers_sizes = std::vector<std::size_t>{};
  layers_sizes.reserve(hidden_layers + 2);
  layers_sizes.push_back(data_supplier_->GetInputLayerSize());
  const auto neurons_per_hidden_layer = static_cast<std::size_t>(
      hyperparameters.at(Index::kNeuronsPerHiddenLayer));
  for (std::size_t i = 0; i < hidden_layers; ++i) {
    layers_sizes.push_back(neurons_per_hidden_layer);
  }
  layers_sizes.push_back(data_supplier_->GetOutputLayerSize());

  auto perceptron = Perceptron(std::move(cost_function),
                               std::move(activation_functions), layers_sizes);
  const auto cfg = SgdConfiguration{
      .epochs = static_cast<std::size_t>(hyperparameters.at(Index::kEpochs)),
      .mini_batch_size =
          static_cast<std::size_t>(hyperparameters.at(Index::kMiniBatchSize)),
      .learning_rate = hyperparameters.at(Index::kLearningRate),
      .monitor_test_accuracy = true,
  };

  const auto train_data = data_supplier_->GetTrainData();
  const auto test_data = data_supplier_->GetTestData();

  auto metrics = perceptron.Sgd(train_data, test_data, cfg);
  return metrics.test_accuracy.back() / test_data.size();
}

}  // namespace nn
