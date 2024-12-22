#include "chromosome.h"

#include <sstream>
#include <stdexcept>

namespace nn {

std::shared_ptr<IChromosome> IChromosome::Create(
    std::vector<double>&& genes, const ChromosomeSubclass subclass) {
  switch (subclass) {
    case ChromosomeSubclass::kSgdHyperparametersKit:
      return std::make_shared<SgdHyperparametersKit>(std::move(genes));
  }
}

const std::vector<double>& IChromosome::get_genes() const { return genes_; }

IChromosome::IChromosome(std::vector<double>&& genes)
    : genes_(std::move(genes)) {}

SgdHyperparametersKit::SgdHyperparametersKit(
    std::vector<double>&& hyperparameters)
    : IChromosome(std::move(hyperparameters)) {
  if (genes_.size() != kHyperparametersNumber) {
    throw std::runtime_error(
        "Got " + std::to_string(genes_.size()) + " SGD hyperparameters, " +
        std::to_string(kHyperparametersNumber) + " expected");
  }
}

double SgdHyperparametersKit::get_learning_rate() const {
  return genes_.at(Index::kLearningRate);
}

std::size_t SgdHyperparametersKit::get_epochs() const {
  return genes_.at(Index::kEpochs);
}

std::size_t SgdHyperparametersKit::get_mini_batch_size() const {
  return genes_.at(Index::kMiniBatchSize);
}

std::size_t SgdHyperparametersKit::get_hidden_layers() const {
  return genes_.at(Index::kHiddenLayers);
}

std::size_t SgdHyperparametersKit::get_neurons_per_hidden_layer() const {
  return genes_.at(Index::kNeuronsPerHiddenLayer);
}

std::string SgdHyperparametersKit::ToString() const {
  auto oss = std::ostringstream{};
  oss << "- Learning rate: " << get_learning_rate()
      << ";\n- Epochs: " << get_epochs()
      << ";\n- Mini-batch size: " << get_mini_batch_size()
      << ";\n- Hidden layers: " << get_hidden_layers()
      << ";\n- Neurons per hidden layer: " << get_neurons_per_hidden_layer();
  return oss.str();
}

}  // namespace nn
