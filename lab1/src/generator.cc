#include "generator.h"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace lab1 {

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
      const IPixelModifier& modifier) const {
    const auto indices_powerset = GeneratePowerset(indices_to_modify_);

    auto modifications = std::vector<std::shared_ptr<const Symbol>>{};
    modifications.reserve(indices_powerset.size());

    for (auto&& indices : indices_powerset) {
      auto variation = sample_;

      for (auto&& index : indices) {
        auto& value = variation.scan(index);
        value = modifier.Modify(value);
      }

      // TODO: Does emplace_back work here?
      modifications.push_back(
          std::make_shared<const Symbol>(std::move(variation)));
    }

    return modifications;
  }
};

const std::array kSymbolModifiers{
    SymbolModifier{{{
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    L'0'},
                   {0, 3, 16, 19}},
    SymbolModifier{{{
                        0, 0, 1, 0,  //
                        0, 1, 1, 0,  //
                        0, 0, 1, 0,  //
                        0, 0, 1, 0,  //
                        0, 1, 1, 1,  //
                    },
                    L'1'},
                   {2, 5, 17, 19}},
    SymbolModifier{{{
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        1, 0, 0, 0,  //
                        1, 1, 1, 1,  //
                    },
                    L'2'},
                   {3, 8, 11, 16}},
    SymbolModifier{{{
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    L'3'},
                   {3, 8, 11, 19}},
    SymbolModifier{{{
                        1, 0, 0, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        0, 0, 0, 1,  //
                    },
                    L'4'},
                   {0, 3, 8, 11}},
    SymbolModifier{{{
                        1, 1, 1, 1,  //
                        1, 0, 0, 0,  //
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    L'5'},
                   {0, 8, 11, 19}},
    SymbolModifier{{{
                        1, 1, 1, 1,  //
                        1, 0, 0, 0,  //
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    L'6'},
                   {0, 11, 16, 19}},
    SymbolModifier{{{
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        0, 0, 1, 0,  //
                        0, 1, 0, 0,  //
                        1, 0, 0, 0,  //
                    },
                    L'7'},
                   {9, 11, 14, 17}},
    SymbolModifier{{{
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    L'8'},
                   {0, 3, 16, 19}},
    SymbolModifier{{{
                        1, 1, 1, 1,  //
                        1, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                        0, 0, 0, 1,  //
                        1, 1, 1, 1,  //
                    },
                    L'9'},
                   {0, 3, 8, 19}},
};

}  // namespace

DataSupplier::Parametrization::Parametrization() {
  const auto pixel_reverser = PixelReverser{};
  for (auto&& symbol_modifier : kSymbolModifiers) {
    auto modifications = symbol_modifier.GenerateModifications(pixel_reverser);
    const auto modifications_size = modifications.size();

    std::shuffle(modifications.begin(), modifications.end(), generator);
    auto begin = std::make_move_iterator(modifications.begin());
    auto error = 0.0;
    for (auto&& [dataset, ratio] : {
             std::make_pair(&training, kTrainingRatio),
             std::make_pair(&validation, kValidationRatio),
             std::make_pair(&testing, kTestingRatio),
         }) {
      double part_size;
      error += std::modf(modifications_size * ratio, &part_size);
      if (error >= 1) {
        --error;
        ++part_size;
      }

      auto end = begin + part_size;
      dataset->reserve(dataset->capacity() + part_size);
      dataset->insert(dataset->end(), begin, end);
      begin = std::move(end);
    }
  }

  for (auto&& dataset : {training, validation, testing}) {
    std::shuffle(dataset.begin(), dataset.end(), generator);
  }
}

}  // namespace lab1
