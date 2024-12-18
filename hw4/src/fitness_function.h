#pragma once

#include <memory>

#include "perceptron.h"

namespace nn {

class IChromosome;

class IFitnessFunction {
 public:
  virtual ~IFitnessFunction() = default;

  virtual double Assess(const IChromosome& chromosome) const = 0;
};

class SgdTestDataCost final : public IFitnessFunction {
  std::unique_ptr<IDataSupplier> data_supplier_;

 public:
  SgdTestDataCost(std::unique_ptr<IDataSupplier>&& data_supplier);

  double Assess(const IChromosome& chromosome) const override;
};

class SgdNagTestDataCost final : public IFitnessFunction {
  std::unique_ptr<IDataSupplier> data_supplier_;

 public:
  SgdNagTestDataCost(std::unique_ptr<IDataSupplier>&& data_supplier);

  double Assess(const IChromosome& chromosome) const override;
};

}  // namespace nn
