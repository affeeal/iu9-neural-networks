#include "genetic_algorithm.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

#include "chromosome.h"

namespace nn {

Segment::Segment(const double left, const double right)
    : left_(left), right_(right) {
  if (left_ >= right_) {
    throw std::runtime_error(
        "The left border must be strictly less than the right one");
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
  const auto fitness_values = CalculateFitnessValue();

  const auto average_fitness_value =
      std::reduce(fitness_values.cbegin(), fitness_values.cend()) /
      fitness_values.size();
  spdlog::info("Population average fitness value: {}", average_fitness_value);

  auto partial_sum = std::vector<double>(cfg_.population_size);
  std::partial_sum(fitness_values.cbegin(), fitness_values.cend(),
                   partial_sum.begin());

  auto partial_sum_to_chromosome =
      std::map<double, std::shared_ptr<IChromosome>>{};
  for (std::size_t i = 0; i < cfg_.population_size; ++i) {
    partial_sum_to_chromosome.insert({partial_sum[i], population_[i]});
  }

  auto selected_chromosomes = std::vector<std::shared_ptr<IChromosome>>{};
  selected_chromosomes.reserve(cfg_.population_size);
  auto distribution =
      std::uniform_real_distribution<double>{0, partial_sum.back()};
  for (std::size_t i = 0; i < cfg_.population_size; ++i) {
    const auto value = distribution(engine_);
    const auto it = partial_sum_to_chromosome.upper_bound(value);
    assert(it != partial_sum_to_chromosome.end());
    selected_chromosomes.push_back(it->second);
  }

  return selected_chromosomes;
}

void GeneticAlgorithm::Crossover(
    std::vector<std::shared_ptr<IChromosome>>& population) {
  static const auto parents_number = static_cast<std::size_t>(
      cfg_.crossover_proportion * cfg_.population_size);
  static auto distribution = std::uniform_real_distribution<>{0.0, 1.0};
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
  static const auto mutants_number =
      static_cast<std::size_t>(cfg_.mutation_proportion * cfg_.population_size);
  static auto distribution =
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
    fitness_values.push_back(fitness_function_->Assess(*population_[i]));
    spdlog::info("Chromosome {}/{} fittness value: {}", i + 1,
                 cfg_.population_size, fitness_values.back());
  }
  return fitness_values;
}

}  // namespace nn
