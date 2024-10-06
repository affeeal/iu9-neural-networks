#include "generator.h"

#include "single_layer_perceptron.h"

namespace lab1 {

std::vector<LabeledData> MarkSymbols(std::vector<Symbol>&& symbols,
                                     const OutputVector output) {
  auto labeled_data = std::vector<LabeledData>{};
  labeled_data.reserve(symbols.size());

  for (auto&& input : symbols) {
    labeled_data.push_back({std::move(input), output});
  }

  return labeled_data;
}

}  // namespace lab1
