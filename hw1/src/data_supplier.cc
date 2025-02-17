#include "data_supplier.h"

#include <cassert>
#include <cmath>
#include <cstdlib>

#include "util.h"

namespace hw1 {

namespace {

class IPixelModifier {
 public:
  ~IPixelModifier() = default;

 public:
  virtual double Modify(const double value) const = 0;
};

class PixelReverser final : public IPixelModifier {
 public:
  double Modify(const double value) const override {
    assert(value == 0 || value == 1);
    return !value;
  }
};

class SymbolModifier final {
  Symbol sample_;
  std::vector<std::size_t> indices_to_modify_;

 public:
  SymbolModifier(Symbol&& sample, std::vector<std::size_t>&& indices_to_modify)
      : sample_(std::move(sample)),
        indices_to_modify_(std::move(indices_to_modify)) {}

  std::vector<std::shared_ptr<const Symbol>> GenerateModifications(
      const IPixelModifier& pixel_modifier) const {
    const auto indices_powerset = GeneratePowerset(indices_to_modify_);

    auto modifications = std::vector<std::shared_ptr<const Symbol>>{};
    modifications.reserve(indices_powerset.size());

    for (auto&& indices : indices_powerset) {
      auto variation = sample_;

      for (auto&& index : indices) {
        auto& value = variation.scan(index);
        value = pixel_modifier.Modify(value);
      }

      modifications.push_back(
          std::make_shared<const Symbol>(std::move(variation)));
    }

    return modifications;
  }
};

const std::array kSymbolModifiers{
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    "0"},
                   {0, 3, 16, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        0, 0, 1, 0,  //
                        0, 1, 1, 0,  //
                        0, 0, 1, 0,  //
                        0, 0, 1, 0,  //
                        0, 1, 1, 1,  //
                    },
                    "1"},
                   {2, 5, 17, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        1, 0, 0, 0,  //
                        1, 1, 1, 1,  //
                    },
                    "2"},
                   {3, 8, 11, 16}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    "3"},
                   {3, 8, 11, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        0, 0, 0, 1,  //
                    },
                    "4"},
                   {0, 3, 8, 11}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        1, 0, 0, 0,  //
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    "5"},
                   {0, 8, 11, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        1, 0, 0, 0,  //
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    "6"},
                   {0, 11, 16, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        0, 0, 1, 0,  //
                        0, 1, 0, 0,  //
                        1, 0, 0, 0,  //
                    },
                    "7"},
                   {9, 11, 14, 17}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    "8"},
                   {0, 3, 16, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    "9"},
                   {0, 3, 8, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        0, 1, 1, 0,  //
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                    },
                    "А"},
                   {4, 5, 6, 7}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        1, 0, 0, 0,  //
                        1, 1, 1, 1,  //
                        1, 0, 0, 0,  //
                        1, 1, 1, 1,  //
                    },
                    "Е"},
                   {0, 8, 11, 16}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 0, 0, 1,  //
                        1, 0, 1, 0,  //
                        1, 1, 0, 0,  //
                        1, 0, 1, 0,  //
                        1, 0, 0, 1,  //
                    },
                    "К"},
                   {0, 3, 16, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                    },
                    "Н"},
                   {0, 3, 16, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                    },
                    "П"},
                   {0, 3, 16, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        1, 0, 0, 0,  //
                        1, 0, 0, 0,  //
                    },
                    "Р"},
                   {0, 3, 11, 16}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        1, 0, 0, 0,  //
                        1, 0, 0, 0,  //
                        1, 0, 0, 0,  //
                        1, 1, 1, 1,  //
                    },
                    "С"},
                   {0, 3, 16, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    "У"},
                   {0, 3, 8, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 0, 0, 0,  //
                        1, 0, 0, 0,  //
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    "Ь"},
                   {0, 11, 16, 19}},
    SymbolModifier{{Eigen::Vector<double, kScanSize>{
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        0, 1, 0, 1,  //
                        1, 0, 0, 1,  //
                    },
                    "Я"},
                   {0, 3, 8, 14}},
};

}  // namespace

struct DataSupplier::Parametrization final {
  static constexpr double kTrainingRatio = 0.5;
  static constexpr double kValidationRatio = 0.2;
  static constexpr double kTestingRatio = 0.3;
  static constexpr std::array kLabels{"0", "1", "2", "3", "4", "5", "6",
                                      "7", "8", "9", "А", "Е", "К", "Н",
                                      "П", "Р", "С", "У", "Ь", "Я"};
  static constexpr std::size_t kLabelsNumber = kLabels.size();

  std::vector<std::shared_ptr<const Symbol>> training;
  std::vector<std::shared_ptr<const Symbol>> validation;
  std::vector<std::shared_ptr<const Symbol>> testing;

  static const Parametrization& GetInstance() {
    static Parametrization parametrization{};
    return parametrization;
  }

  Parametrization(const Parametrization& other) = delete;
  Parametrization& operator=(const Parametrization& other) = delete;

 private:
  std::default_random_engine generator;

  Parametrization();
};

const DataSupplier::Parametrization& DataSupplier::kParametrization =
    DataSupplier::Parametrization::GetInstance();

constexpr double kEpsilon = 10e-6;

bool AreEqual(const double x, const double y, const double epsion) {
  return std::abs(x - y) < epsion;
}

DataSupplier::Parametrization::Parametrization() {
  const auto pixel_reverser = PixelReverser{};
  for (auto&& symbol_modifier : kSymbolModifiers) {
    auto modifications = symbol_modifier.GenerateModifications(pixel_reverser);
    const auto modifications_size = modifications.size();

    std::shuffle(modifications.begin(), modifications.end(), generator);
    auto begin = std::make_move_iterator(modifications.begin());
    auto error = 0.0;
    for (auto&& [dataset, ratio] : {
             std::make_pair(std::reference_wrapper(training), kTrainingRatio),
             std::make_pair(std::reference_wrapper(validation),
                            kValidationRatio),
             std::make_pair(std::reference_wrapper(testing), kTestingRatio),
         }) {
      double part_size;
      error += std::modf(modifications_size * ratio, &part_size);
      if (AreEqual(error, 1, kEpsilon)) {
        --error;
        ++part_size;
      }

      auto end = begin + part_size;
      dataset.reserve(dataset.capacity() + part_size);
      dataset.insert(dataset.end(), begin, end);
      begin = std::move(end);
    }
  }

  for (auto&& [begin, end] :
       {std::make_pair(training.begin(), training.end()),
        std::make_pair(validation.begin(), validation.end()),
        std::make_pair(testing.begin(), testing.end())}) {
    std::shuffle(begin, end, generator);
  }
}

DataSupplier::DataSupplier(const double low_score, const double high_score) {
  auto label_to_y =
      std::unordered_map<std::string, std::shared_ptr<const Eigen::VectorXd>>{};
  label_to_y.reserve(Parametrization::kLabelsNumber);
  for (std::size_t i = 0; i < Parametrization::kLabelsNumber; ++i) {
    auto y = Eigen::VectorXd(Parametrization::kLabelsNumber);
    y.fill(low_score);
    y(i) = high_score;
    label_to_y.insert({Parametrization::kLabels[i],
                       std::make_shared<const Eigen::VectorXd>(std::move(y))});
  }

  for (auto&& [dataset, symbol_dataset] :
       {std::make_pair(std::reference_wrapper(training_),
                       std::reference_wrapper(kParametrization.training)),
        std::make_pair(std::reference_wrapper(validation_),
                       std::reference_wrapper(kParametrization.validation)),
        std::make_pair(std::reference_wrapper(testing_),
                       std::reference_wrapper(kParametrization.testing))}) {
    dataset.reserve(symbol_dataset.size());
    for (auto&& symbol : symbol_dataset) {
      dataset.push_back(
          std::make_shared<const Data>(symbol, label_to_y.at(symbol->label)));
    }
  }
}

}  // namespace hw1
