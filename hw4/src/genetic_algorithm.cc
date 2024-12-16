#include "genetic_algorithm.h"

#include <spdlog/spdlog.h>

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
    const std::size_t populations_number, const std::size_t population_size)
    : random_engine_(std::random_device{}()),
      fitness_function_(std::move(fitness_function)),
      subclass_(subclass),
      populations_number_(populations_number),
      population_size_(population_size) {
  const auto genes_number = segments.size();
  distributions_.reserve(genes_number);
  for (auto&& segment : segments) {
    distributions_.push_back(std::uniform_real_distribution<>(
        segment.get_left(), segment.get_right()));
  }

  population_.reserve(population_size);
  for (std::size_t i = 0; i < population_size; ++i) {
    auto genes = std::vector<double>{};
    genes.reserve(genes_number);
    for (auto&& distribution : distributions_) {
      genes.push_back(distribution(random_engine_));
    }
    population_.push_back(IChromosome::Create(std::move(genes), subclass_));
  }
}

std::shared_ptr<IChromosome> GeneticAlgorithm::Run() {
  for (std::size_t i = 1; i <= populations_number_; ++i) {
    const auto new_population = RouletteWheelSelection();
    break;
    // TODO
  }

  return nullptr;  // TODO
}

std::vector<std::shared_ptr<IChromosome>>
GeneticAlgorithm::RouletteWheelSelection() {
  auto fitness_values = std::vector<double>{};
  fitness_values.reserve(population_size_);
  for (auto&& chromosome : population_) {
    fitness_values.push_back(fitness_function_->Assess(*chromosome));
  }

  auto partial_sum = std::vector<double>(population_size_);
  std::partial_sum(fitness_values.cbegin(), fitness_values.cend(),
                   partial_sum.begin());

  auto partial_sum_to_chromosome =
      std::map<double, std::shared_ptr<IChromosome>>{};
  for (std::size_t i = 0; i < population_size_; ++i) {
    partial_sum_to_chromosome.insert({partial_sum[i], population_[i]});
  }

  auto new_population = std::vector<std::shared_ptr<IChromosome>>{};
  new_population.reserve(population_size_);
  auto distribution = std::uniform_real_distribution<>{0, partial_sum.back()};
  for (std::size_t i = 0; i < population_size_; ++i) {
    const auto value = distribution(random_engine_);
    const auto it = partial_sum_to_chromosome.upper_bound(value);
    assert(it != partial_sum_to_chromosome.end());
    new_population.push_back(it->second);
  }

  return new_population;
}

}  // namespace nn
