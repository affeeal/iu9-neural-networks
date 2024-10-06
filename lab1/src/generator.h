#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "data_supplier.h"
#include "single_layer_perceptron.h"
#include "util.h"

namespace lab1 {

// struct Symbol final {
//   InputVector vector_;
//   wchar_t representation_;
// };
//
// struct LabeledData final {
//   Symbol input;
//   OutputVector output;
// };
//
// class VariationsGenerator final {
//   Symbol sample_;
//   std::vector<std::size_t> indices_to_modify_;
//
//  public:
//   VariationsGenerator(Symbol&& sample,
//                       std::vector<std::size_t>&& indices_to_modify)
//       : sample_(std::move(sample)),
//         indices_to_modify_(std::move(indices_to_modify)) {}
//
//   std::vector<Symbol> GetVariations() const {
//     const auto indices_powerset = GetPowerset(indices_to_modify_);
//
//     auto variations = std::vector<Symbol>{};
//     variations.reserve(indices_powerset.size());
//
//     for (auto&& indices : indices_powerset) {
//       auto variation = sample_;
//
//       for (auto&& index : indices) {
//         auto& value = variation.vector_(index);
//         assert(value == 0 || value == 1);
//         value = !value;
//       }
//
//       variations.push_back(std::move(variation));
//     }
//
//     return variations;
//   }
// };
//
// const std::array<VariationsGenerator, kSymbolsCount> kGenerators{
//     VariationsGenerator{{{
//                              1, 1, 1, 1,  //
//                              1, 0, 0, 1,  //
//                              1, 0, 0, 1,  //
//                              1, 0, 0, 1,  //
//                              1, 1, 1, 1,  //
//                          },
//                          L'0'},
//                         {0, 3, 16, 19}},
//     VariationsGenerator{{{
//                              0, 0, 1, 0,  //
//                              0, 1, 1, 0,  //
//                              0, 0, 1, 0,  //
//                              0, 0, 1, 0,  //
//                              0, 1, 1, 1,  //
//                          },
//                          L'1'},
//                         {2, 5, 17, 19}},
//     VariationsGenerator{{{
//                              1, 1, 1, 1,  //
//                              0, 0, 0, 1,  //
//                              1, 1, 1, 1,  //
//                              1, 0, 0, 0,  //
//                              1, 1, 1, 1,  //
//                          },
//                          L'2'},
//                         {3, 8, 11, 16}},
//     VariationsGenerator{{{
//                              1, 1, 1, 1,  //
//                              0, 0, 0, 1,  //
//                              1, 1, 1, 1,  //
//                              0, 0, 0, 1,  //
//                              1, 1, 1, 1,  //
//                          },
//                          L'3'},
//                         {3, 8, 11, 19}},
//     VariationsGenerator{{{
//                              1, 0, 0, 1,  //
//                              1, 0, 0, 1,  //
//                              1, 1, 1, 1,  //
//                              0, 0, 0, 1,  //
//                              0, 0, 0, 1,  //
//                          },
//                          L'4'},
//                         {0, 3, 8, 11}},
//     VariationsGenerator{{{
//                              1, 1, 1, 1,  //
//                              1, 0, 0, 0,  //
//                              1, 1, 1, 1,  //
//                              0, 0, 0, 1,  //
//                              1, 1, 1, 1,  //
//                          },
//                          L'5'},
//                         {0, 8, 11, 19}},
//     VariationsGenerator{{{
//                              1, 1, 1, 1,  //
//                              1, 0, 0, 0,  //
//                              1, 1, 1, 1,  //
//                              1, 0, 0, 1,  //
//                              1, 1, 1, 1,  //
//                          },
//                          L'6'},
//                         {0, 11, 16, 19}},
//     VariationsGenerator{{{
//                              1, 1, 1, 1,  //
//                              0, 0, 0, 1,  //
//                              0, 0, 1, 0,  //
//                              0, 1, 0, 0,  //
//                              1, 0, 0, 0,  //
//                          },
//                          L'7'},
//                         {9, 11, 14, 17}},
//     VariationsGenerator{{{
//                              1, 1, 1, 1,  //
//                              1, 0, 0, 1,  //
//                              1, 1, 1, 1,  //
//                              1, 0, 0, 1,  //
//                              1, 1, 1, 1,  //
//                          },
//                          L'8'},
//                         {0, 3, 16, 19}},
//     VariationsGenerator{{{
//                              1, 1, 1, 1,  //
//                              1, 0, 0, 1,  //
//                              1, 1, 1, 1,  //
//                              0, 0, 0, 1,  //
//                              1, 1, 1, 1,  //
//                          },
//                          L'9'},
//                         {0, 3, 8, 19}},
// };
//
// std::vector<LabeledData> MarkSymbols(std::vector<Symbol>&& symbols,
//                                      const OutputVector output);
//
// template <typename Iter>
// // requires std::input_iterator<Iter>
// std::vector<LabeledData> GenerateData(const Iter begin, const Iter end) {
//   for (auto it = begin; it != end; ++it) {
//     auto&& generator = *it;
//     auto variations = generator.GetVariations();
//     auto labeled_data = MarkSymbols(variations, output);
//   }
// }
//
// class LabeledDataGenerator final {
//   struct Parametrization final {
//     std::array<wchar_t, kSymbolsCount> labels;
//     std::unordered_map<wchar_t, OutputVector> label_to_output_vector;
//
//     Parametrization()
//         : labels{L'0', L'1', L'2', L'3', L'4', L'5', L'6', L'7', L'8', L'9'}
//         {
//       label_to_output_vector.reserve(labels.size());
//       for (std::size_t i = 0; i < kSymbolsCount; ++i) {
//         label_to_output_vector.insert({labels[i], OutputVector::Unit(i)});
//       }
//     }
//   };
//
//   static const Parametrization kParametrization;
// };

constexpr std::size_t kSymbolScanSize = 20;  // NOTE: Scan means "развёртка"
constexpr std::size_t kSymbolsCount = 10;  // TODO: Increase to 20

using InputVector = Eigen::Vector<double, kSymbolScanSize>;
using OutputVector = Eigen::Vector<double, kSymbolsCount>;

class Data final : public IData {
  InputVector input_;
  OutputVector output_;

 public:
  Eigen::VectorXd GetX() const override { return input_; }

  Eigen::VectorXd GetY() const override { return output_; }
};

class DataSupplier final : public IDataSupplier {
  std::vector<std::shared_ptr<const IData>> training_;
  std::vector<std::shared_ptr<const IData>> validation_;
  std::vector<std::shared_ptr<const IData>> testing_;

 public:
  static DataSupplier& GetInstance() {
    static DataSupplier data_supplier;
    return data_supplier;
  }

  std::vector<std::shared_ptr<const IData>> GetTrainingData() const override {
    return training_;
  }

  std::vector<std::shared_ptr<const IData>> GetValidationData() const override {
    return validation_;
  }

  std::vector<std::shared_ptr<const IData>> GetTestingData() const override {
    return testing_;
  }

 public:
  DataSupplier(const DataSupplier& other) = delete;
  DataSupplier& operator=(const DataSupplier& other) = delete;

 private:
  DataSupplier() = default;
};

}  // namespace lab1
