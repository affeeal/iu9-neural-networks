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

class ISgdFitness : public IFitnessFunction {
 public:
  virtual ~ISgdFitness() = default;

  ISgdFitness(std::unique_ptr<IDataSupplier>&& data_supplier);

 protected:
  std::unique_ptr<IDataSupplier> data_supplier_;
};

class SgdFitness final : public ISgdFitness {
 public:
  using ISgdFitness::ISgdFitness;

  double Assess(const IChromosome& chromosome) const override;
};

class SgdNagFitness final : public ISgdFitness {
 public:
  using ISgdFitness::ISgdFitness;

  double Assess(const IChromosome& chromosome) const override;
};

class SgdAdagradFitness final : public ISgdFitness {
 public:
  using ISgdFitness::ISgdFitness;

  double Assess(const IChromosome& chromosome) const override;
};

class SgdAdamFitness final : public ISgdFitness {
 public:
  using ISgdFitness::ISgdFitness;

  double Assess(const IChromosome& chromosome) const override;
};

}  // namespace nn
