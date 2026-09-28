#pragma once

#include <charconv>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "perceptron.h"

namespace nn {

// MNIST CSV: integer label followed by 784 integer pixels, without a header.
template <typename Data>
std::vector<std::shared_ptr<const IData>> ReadMnistCsv(
    const std::string& filename, double false_score, double true_score) {
  if (!std::isfinite(false_score) || !std::isfinite(true_score)) {
    throw std::invalid_argument("MNIST target scores must be finite");
  }
  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Cannot open MNIST CSV: " + filename);
  std::vector<std::shared_ptr<const IData>> result;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    Data data;
    data.x.resize(784);
    data.y = Eigen::VectorXd::Constant(10, false_score);
    std::string_view remaining(line);
    for (std::size_t column = 0; column < 785; ++column) {
      const auto comma = remaining.find(',');
      const auto field = remaining.substr(0, comma);
      int value = 0;
      const auto [end, error] =
          std::from_chars(field.data(), field.data() + field.size(), value);
      if (error != std::errc{} || end != field.data() + field.size() ||
          value < 0 || value > (column == 0 ? 9 : 255) ||
          (column < 784 && comma == std::string_view::npos) ||
          (column == 784 && comma != std::string_view::npos)) {
        throw std::runtime_error(filename + ":" + std::to_string(line_number) +
                                 ": invalid MNIST field " +
                                 std::to_string(column + 1));
      }
      if (column == 0) {
        data.label = std::to_string(value);
        data.y[value] = true_score;
      } else {
        data.x[column - 1] = value / 255.0;
      }
      if (comma != std::string_view::npos) remaining.remove_prefix(comma + 1);
    }
    result.push_back(std::make_shared<const Data>(std::move(data)));
  }
  if (file.bad())
    throw std::runtime_error("Cannot read MNIST CSV: " + filename);
  if (result.empty()) throw std::runtime_error("Empty MNIST CSV: " + filename);
  return result;
}

}  // namespace nn
