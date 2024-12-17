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
 public:
  // TODO: Validate the configuration
  struct Configuration final {
    std::size_t populations_number;
    std::size_t population_size;
    double crossover_proportion;
    double mutation_proportion;
  };

 private:
  std::mt19937 engine_{std::random_device{}()};

  std::unique_ptr<IFitnessFunction> fitness_function_;
  ChromosomeSubclass chromosome_subclass_;
  std::vector<std::uniform_real_distribution<double>> genes_distributions_;
  std::vector<std::shared_ptr<IChromosome>> population_;
  Configuration cfg_;
  std::size_t genes_number_;

 public:
  GeneticAlgorithm(std::unique_ptr<IFitnessFunction>&& fitness_function,
                   const ChromosomeSubclass subclass,
                   const std::vector<Segment>& segments,
                   const Configuration& cfg);

  std::shared_ptr<IChromosome> Run();

 private:
  std::vector<std::shared_ptr<IChromosome>> RouletteWheelSelection();
  void Crossover(std::vector<std::shared_ptr<IChromosome>>& population);
  void Mutate(std::vector<std::shared_ptr<IChromosome>>& population);

  std::vector<double> CalculateFitnessValue() const;
};

}  // namespace nn
