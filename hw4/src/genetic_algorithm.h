#pragma once

#include <memory>
#include <random>
#include <vector>

#include "chromosome.h"
#include "fitness_function.h"

namespace nn {

class Segment final {
  double left_, right_;

 public:
  Segment(const double left, const double right);

  double get_left() const;
  double get_right() const;
};

class GeneticAlgorithm final {
  std::mt19937 random_engine_;
  std::unique_ptr<IFitnessFunction> fitness_function_;
  ChromosomeSubclass subclass_;
  std::size_t populations_number_;
  std::size_t population_size_;
  std::vector<std::uniform_real_distribution<double>> distributions_;
  std::vector<std::shared_ptr<IChromosome>> population_;

 public:
  GeneticAlgorithm(std::unique_ptr<IFitnessFunction>&& fitness_function,
                   const ChromosomeSubclass subclass,
                   const std::vector<Segment>& segments,
                   const std::size_t populations_number,
                   const std::size_t population_size);

  std::shared_ptr<IChromosome> Run();

 private:
  std::vector<std::shared_ptr<IChromosome>> RouletteWheelSelection();
};

}  // namespace nn
