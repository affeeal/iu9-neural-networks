#pragma once

#include <memory>

#include "perceptron.h"

namespace nn {

class IChromosome;

class IFitnessFunction {
 public:
  virtual ~IFitnessFunction() = default;

 public:
  virtual double Assess(const IChromosome& chromosome) const = 0;
};

class AccuracyOnTestData final : public IFitnessFunction {
  std::unique_ptr<IDataSupplier> data_supplier_;

 public:
  AccuracyOnTestData(std::unique_ptr<IDataSupplier>&& data_supplier);

  double Assess(const IChromosome& chromosome) const override;
};

}  // namespace nn
