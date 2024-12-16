#pragma once

#include <memory>
#include <random>
#include <vector>

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
  std::size_t populations_number_;
  std::size_t population_size_;
  std::vector<std::uniform_real_distribution<double>> distributions_;
  std::vector<Chromosome> population_;

 public:
  GeneticAlgorithm(std::unique_ptr<IFitnessFunction>&& fitness_function,
                   const std::vector<Segment>& segments,
                   const std::size_t populations_number,
                   const std::size_t population_size);

  Chromosome Run();

 private:
  std::vector<Chromosome> RouletteWheelSelection();
};

}  // namespace nn
