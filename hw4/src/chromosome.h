#pragma once

#include <memory>
#include <vector>

#include "fitness_function.h"

namespace nn {

enum class ChromosomeSubclass {
  kSgdKit,
  kSgdNagKit,
};

class IChromosome {
 public:
  static std::shared_ptr<IChromosome> Create(std::vector<double>&& genes,
                                             const ChromosomeSubclass subclass);

 public:
  virtual ~IChromosome() = default;

  virtual std::string ToString() const = 0;

 public:
  const std::vector<double>& get_genes() const;

 protected:
  IChromosome(std::vector<double>&& genes);

  std::vector<double> genes_;
};

class SgdKit : public IChromosome {
 protected:
  enum Index : std::size_t {
    kLearningRate,
    kEpochs,
    kMiniBatchSize,
    kHiddenLayers,
    kNeuronsPerHiddenLayer,
  };

  static constexpr std::size_t kHyperparametersNumber = 5;

 public:
  virtual ~SgdKit() = default;

  SgdKit(std::vector<double>&& hyperparameters);

  double get_learning_rate() const;
  std::size_t get_epochs() const;
  std::size_t get_mini_batch_size() const;
  std::size_t get_hidden_layers() const;
  std::size_t get_neurons_per_hidden_layer() const;

  std::string ToString() const override;
};

class SgdNagKit final : public SgdKit {
  enum Index : std::size_t {
    kGamma = SgdKit::kHyperparametersNumber,
  };

  static constexpr std::size_t kHyperparametersNumber = 6;

 public:
  SgdNagKit(std::vector<double>&& hyperparameters);

  double get_gamma() const;

  std::string ToString() const override;
};

}  // namespace nn
