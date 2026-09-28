#include <spdlog/spdlog.h>

#include <limits>

#include "check.h"
#include "data_supplier.h"
#include "genetic_algorithm.h"

namespace {
class ConstantFitness final : public nn::IFitnessFunction {
  double value_;

 public:
  explicit ConstantFitness(double value) : value_(value) {}
  double Assess(const nn::IChromosome&) const override { return value_; }
};

class ValidationOnlySupplier final : public nn::IDataSupplier {
  bool divergent_;

 public:
  explicit ValidationOnlySupplier(bool divergent = false)
      : divergent_(divergent) {}
  std::size_t GetInputLayerSize() const override { return 1; }
  std::size_t GetOutputLayerSize() const override { return 2; }
  std::vector<std::shared_ptr<const nn::IData>> GetTrainData() const override {
    auto sample = std::make_shared<nn::Data>();
    sample->x = Eigen::VectorXd::Constant(1, divergent_ ? 1e200 : 0.2);
    sample->y = Eigen::Vector2d(0.5, 0.5);
    return {sample};
  }
  std::vector<std::shared_ptr<const nn::IData>> GetValidationData()
      const override {
    return GetTrainData();
  }
  std::vector<std::shared_ptr<const nn::IData>> GetTestData() const override {
    throw std::runtime_error("Hyperparameter search must not access test data");
  }
};
}  // namespace

int main() {
  spdlog::set_level(spdlog::level::off);
  const std::vector<nn::Segment> segments{
      {0.01, 0.02}, {1, 1}, {1, 1}, {0, 0}, {2, 2}};
  for (const auto population : {1U, 7U}) {
    for (const auto value :
         {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), 1.0}) {
      nn::GeneticAlgorithm algorithm(
          std::make_unique<ConstantFitness>(value),
          nn::ChromosomeSubclass::kSgdHyperparametersKit, segments,
          {2, population, 1, 1});
      const auto result = algorithm.Run();
      CHECK(result && result->get_genes().size() == 5);
    }
  }
  Throws([&] {
    nn::GeneticAlgorithm invalid(std::make_unique<ConstantFitness>(1),
                                 nn::ChromosomeSubclass::kSgdHyperparametersKit,
                                 segments, {1, 0, 0.5, 0.5});
  });
  Throws([&] {
    nn::GeneticAlgorithm invalid(std::make_unique<ConstantFitness>(1),
                                 nn::ChromosomeSubclass::kSgdHyperparametersKit,
                                 segments, {1, 2, 1.1, 0.5});
  });
  Throws([] { nn::Segment invalid(1, 0); });
  Throws(
      [] { nn::Segment invalid(0, std::numeric_limits<double>::infinity()); });
  Throws([] { nn::SgdHyperparametersKit invalid({0.1, 0, 1, 0, 2}); });
  Throws([] { nn::SgdHyperparametersKit invalid({0.1, 1, 1, -1, 2}); });
  Throws([] {
    nn::IChromosome::Create({}, static_cast<nn::ChromosomeSubclass>(10));
  });

  for (bool divergent : {false, true}) {
    const nn::SgdHyperparametersKit kit({divergent ? 1e308 : 0.01, 1, 1, 0, 2});
    std::vector<std::unique_ptr<nn::IFitnessFunction>> fitness;
    fitness.push_back(std::make_unique<nn::SgdFitness>(
        std::make_unique<ValidationOnlySupplier>(divergent)));
    fitness.push_back(std::make_unique<nn::SgdNagFitness>(
        std::make_unique<ValidationOnlySupplier>(divergent)));
    fitness.push_back(std::make_unique<nn::SgdAdagradFitness>(
        std::make_unique<ValidationOnlySupplier>(divergent)));
    fitness.push_back(std::make_unique<nn::SgdAdamFitness>(
        std::make_unique<ValidationOnlySupplier>(divergent)));
    for (const auto& f : fitness) {
      const auto value = f->Assess(kit);
      if (divergent) {
        Near(value, 0);
      } else {
        CHECK(std::isfinite(value) && value > 0 && value <= 1);
      }
    }
  }
}
