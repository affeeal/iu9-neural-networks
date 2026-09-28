#include "genetic_algorithm.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include "chromosome.h"

namespace nn {

Segment::Segment(const double left, const double right)
    : left_(left), right_(right) {
  if (!std::isfinite(left_) || !std::isfinite(right_) || left_ > right_) {
    throw std::runtime_error(
        "The left border must be less or equal than the right one");
  }
}

double Segment::get_left() const { return left_; }
double Segment::get_right() const { return right_; }

GeneticAlgorithm::GeneticAlgorithm(
    std::unique_ptr<IFitnessFunction>&& fitness_function,
    const ChromosomeSubclass subclass, const std::vector<Segment>& segments,
    const GeneticAlgorithm::Configuration& cfg)
    : fitness_function_(std::move(fitness_function)),
      chromosome_subclass_(subclass),
      cfg_(cfg),
      genes_number_(segments.size()) {
  const auto proportion_valid = [](double p) {
    return std::isfinite(p) && p >= 0 && p <= 1;
  };
  if (!fitness_function_ || cfg.population_size == 0 ||
      genes_number_ != SgdHyperparametersKit::kHyperparametersNumber ||
      !proportion_valid(cfg.crossover_proportion) ||
      !proportion_valid(cfg.mutation_proportion)) {
    throw std::invalid_argument("Invalid genetic algorithm configuration");
  }
  genes_distributions_.reserve(genes_number_);
  for (auto&& segment : segments) {
    genes_distributions_.push_back(std::uniform_real_distribution<>(
        segment.get_left(), segment.get_right()));
  }

  population_.reserve(cfg.population_size);
  for (std::size_t i = 0; i < cfg.population_size; ++i) {
    auto genes = std::vector<double>{};
    genes.reserve(genes_number_);
    for (auto&& distribution : genes_distributions_) {
      genes.push_back(distribution(engine_));
    }
    population_.push_back(
        IChromosome::Create(std::move(genes), chromosome_subclass_));
  }
}

std::shared_ptr<IChromosome> GeneticAlgorithm::Run() {
  for (std::size_t i = 0; i < cfg_.populations_number; ++i) {
    spdlog::info("Population {}/{}:", i, cfg_.populations_number);
    for (std::size_t j = 0; j < cfg_.population_size; ++j) {
      spdlog::info("Chromosome {}/{}:\n{}", j + 1, cfg_.population_size,
                   population_[j]->ToString());
    }

    auto new_population = RouletteWheelSelection();
    Crossover(new_population);
    std::shuffle(new_population.begin(), new_population.end(), engine_);
    Mutate(new_population);
    std::shuffle(new_population.begin(), new_population.end(), engine_);

    population_ = std::move(new_population);
  }

  spdlog::info("Population {}/{}:", cfg_.populations_number,
               cfg_.populations_number);
  for (std::size_t j = 0; j < cfg_.population_size; ++j) {
    spdlog::info("Chromosome {}/{}:\n{}", j + 1, cfg_.population_size,
                 population_[j]->ToString());
  }

  const auto fitness_values = CalculateFitnessValue();
  const auto fittest_chromosome_index = std::distance(
      fitness_values.cbegin(),
      std::max_element(fitness_values.cbegin(), fitness_values.cend()));

  spdlog::info("Chromosome {} (the fittest one):\n{}",
               fittest_chromosome_index + 1,
               population_[fittest_chromosome_index]->ToString());
  return population_[fittest_chromosome_index];
}

std::vector<std::shared_ptr<IChromosome>>
GeneticAlgorithm::RouletteWheelSelection() {
  auto fitness_values = CalculateFitnessValue();
  const double largest =
      *std::max_element(fitness_values.begin(), fitness_values.end());
  for (auto& value : fitness_values) value = largest == 0 ? 1 : value / largest;
  std::discrete_distribution<std::size_t> distribution(fitness_values.begin(),
                                                       fitness_values.end());
  std::vector<std::shared_ptr<IChromosome>> selected;
  selected.reserve(cfg_.population_size);
  for (std::size_t i = 0; i < cfg_.population_size; ++i) {
    selected.push_back(population_[distribution(engine_)]);
  }
  return selected;
}

void GeneticAlgorithm::Crossover(
    std::vector<std::shared_ptr<IChromosome>>& population) {
  const auto parents_number = static_cast<std::size_t>(
      cfg_.crossover_proportion * cfg_.population_size);
  auto distribution = std::uniform_real_distribution<>{0.0, 1.0};
  for (std::size_t i = 0; i + 1 < parents_number; i += 2) {
    const auto alpha = distribution(engine_);

    const auto& parent1_genes = population[i]->get_genes();
    const auto& parent2_genes = population[i + 1]->get_genes();

    auto offspring1_genes = std::vector<double>{};
    offspring1_genes.reserve(genes_number_);
    for (std::size_t j = 0; j < genes_number_; ++j) {
      offspring1_genes.push_back(alpha * parent1_genes[j] +
                                 (1 - alpha) * parent2_genes[j]);
    }

    auto offspring2_genes = std::vector<double>{};
    offspring2_genes.reserve(genes_number_);
    for (std::size_t j = 0; j < genes_number_; ++j) {
      offspring2_genes.push_back((1 - alpha) * parent1_genes[j] +
                                 alpha * parent2_genes[j]);
    }

    population[i] =
        IChromosome::Create(std::move(offspring1_genes), chromosome_subclass_);
    population[i + 1] =
        IChromosome::Create(std::move(offspring2_genes), chromosome_subclass_);
  }
}

void GeneticAlgorithm::Mutate(
    std::vector<std::shared_ptr<IChromosome>>& population) {
  const auto mutants_number =
      static_cast<std::size_t>(cfg_.mutation_proportion * cfg_.population_size);
  auto distribution =
      std::uniform_int_distribution<>{0, static_cast<int>(genes_number_) - 1};
  for (std::size_t i = 0; i < mutants_number; ++i) {
    const auto mutated_gene_index = distribution(engine_);

    auto genes = population[i]->get_genes();
    genes[mutated_gene_index] =
        genes_distributions_[mutated_gene_index](engine_);
    population[i] = IChromosome::Create(std::move(genes), chromosome_subclass_);
  }
}

std::vector<double> GeneticAlgorithm::CalculateFitnessValue() const {
  auto fitness_values = std::vector<double>{};
  fitness_values.reserve(cfg_.population_size);
  for (std::size_t i = 0; i < cfg_.population_size; ++i) {
    const double fitness = fitness_function_->Assess(*population_[i]);
    fitness_values.push_back(std::isfinite(fitness) && fitness > 0 ? fitness
                                                                   : 0);
    spdlog::info("Chromosome {}/{} fitness value: {}", i + 1,
                 cfg_.population_size, fitness_values.back());
  }
  return fitness_values;
}

}  // namespace nn
