#pragma once

#include <memory>

#include "perceptron.h"

namespace nn {

static constexpr std::size_t kHyperparametersNumber = 5;

enum Index : std::size_t {
  kLearningRate,
  kEpochs,
  kMiniBatchSize,
  kHiddenLayers,
  kNeuronsPerHiddenLayer,
};

class Chromosome final {
 public:
  Chromosome(std::vector<double>&& genes);

  const std::vector<double>& get_genes() const;

  std::vector<double> genes_;
};

class IFitnessFunction {
 public:
  virtual ~IFitnessFunction() = default;

 public:
  virtual double Assess(const Chromosome& chromosome) const = 0;
};

class AccuracyOnTestData final : public IFitnessFunction {
  std::unique_ptr<IDataSupplier> data_supplier_;

 public:
  AccuracyOnTestData(std::unique_ptr<IDataSupplier>&& data_supplier);

  double Assess(const Chromosome& chromosome) const override;
};

}  // namespace nn
