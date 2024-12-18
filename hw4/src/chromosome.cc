#include "chromosome.h"

#include <sstream>
#include <stdexcept>

namespace nn {

std::shared_ptr<IChromosome> IChromosome::Create(
    std::vector<double>&& genes, const ChromosomeSubclass subclass) {
  switch (subclass) {
    case ChromosomeSubclass::kSgdKit:
      return std::make_shared<SgdKit>(std::move(genes));
    case nn::ChromosomeSubclass::kSgdNagKit:
      return std::make_shared<SgdNagKit>(std::move(genes));
  }
}

const std::vector<double>& IChromosome::get_genes() const { return genes_; }

IChromosome::IChromosome(std::vector<double>&& genes)
    : genes_(std::move(genes)) {}

SgdKit::SgdKit(std::vector<double>&& hyperparameters)
    : IChromosome(std::move(hyperparameters)) {
  if (genes_.size() < kHyperparametersNumber) {
    throw std::runtime_error("Got " + std::to_string(genes_.size()) +
                             " SGD hyperparameters, at least " +
                             std::to_string(kHyperparametersNumber) +
                             " expected");
  }
}

double SgdKit::get_learning_rate() const {
  return genes_.at(Index::kLearningRate);
}

std::size_t SgdKit::get_epochs() const { return genes_.at(Index::kEpochs); }

std::size_t SgdKit::get_mini_batch_size() const {
  return genes_.at(Index::kMiniBatchSize);
}

std::size_t SgdKit::get_hidden_layers() const {
  return genes_.at(Index::kHiddenLayers);
}

std::size_t SgdKit::get_neurons_per_hidden_layer() const {
  return genes_.at(Index::kNeuronsPerHiddenLayer);
}

std::string SgdKit::ToString() const {
  auto oss = std::ostringstream{};
  oss << "- Learning rate: " << get_learning_rate()
      << ";\n- Epochs: " << get_epochs()
      << ";\n- Mini-batch size: " << get_mini_batch_size()
      << ";\n- Hidden layers: " << get_hidden_layers()
      << ";\n- Neurons per hidden layer: " << get_neurons_per_hidden_layer();
  return oss.str();
}

SgdNagKit::SgdNagKit(std::vector<double>&& hyperparameters)
    : SgdKit(std::move(hyperparameters)) {
  if (genes_.size() != kHyperparametersNumber) {
    throw std::runtime_error(
        "Got " + std::to_string(genes_.size()) + " SGD hyperparameters, " +
        std::to_string(kHyperparametersNumber) + " expected");
  }
}

double SgdNagKit::get_gamma() const { return genes_.at(Index::kGamma); }

std::string SgdNagKit::ToString() const {
  return SgdKit::ToString() + ";\n- Gamma: " + std::to_string(get_gamma());
}

}  // namespace nn
