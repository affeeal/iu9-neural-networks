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
      subclass_(subclass),
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
    population_.push_back(IChromosome::Create(std::move(genes), subclass_));
  }
}

std::shared_ptr<IChromosome> GeneticAlgorithm::Run() {
  for (std::size_t i = 1; i <= cfg_.populations_number; ++i) {
    auto chromosomes = RouletteWheelSelection();
    Crossover(chromosomes);
    std::shuffle(chromosomes.begin(), chromosomes.end(), engine_);
  }

  return nullptr;  // TODO
}

std::vector<std::shared_ptr<IChromosome>>
GeneticAlgorithm::RouletteWheelSelection() const {
  auto fitness_values = std::vector<double>{};
  fitness_values.reserve(cfg_.population_size);
  for (auto&& chromosome : population_) {
    fitness_values.push_back(fitness_function_->Assess(*chromosome));
  }

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
  auto distribution = std::uniform_real_distribution<>{0, partial_sum.back()};
  for (std::size_t i = 0; i < cfg_.population_size; ++i) {
    const auto value = distribution(engine_);
    const auto it = partial_sum_to_chromosome.upper_bound(value);
    assert(it != partial_sum_to_chromosome.end());
    selected_chromosomes.push_back(it->second);
  }

  return selected_chromosomes;
}

void GeneticAlgorithm::Crossover(
    std::vector<std::shared_ptr<IChromosome>>& population) const {
  const auto parents_number = static_cast<std::size_t>(
      cfg_.crossover_probability * cfg_.population_size);
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

    population[i] = IChromosome::Create(std::move(offspring1_genes), subclass_);
    population[i + 1] =
        IChromosome::Create(std::move(offspring2_genes), subclass_);
  }
}

}  // namespace nn
