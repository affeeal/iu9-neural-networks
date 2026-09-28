#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

namespace hw1 {

template <typename T>
std::vector<std::vector<T>> GeneratePowerset(const std::vector<T>& set) {
  const auto set_size = set.size();
  if (set_size >= std::numeric_limits<std::size_t>::digits) {
    throw std::length_error("Powerset size is not representable");
  }
  const auto powerset_size = std::size_t{1} << set_size;

  auto powerset = std::vector<std::vector<T>>{};
  powerset.reserve(powerset_size);

  for (std::size_t i = 0; i < powerset_size; ++i) {
    auto subset = std::vector<T>{};

    for (std::size_t j = 0; j < set_size; ++j) {
      if (i & (std::size_t{1} << j)) {
        subset.push_back(set.at(j));
      }
    }

    powerset.push_back(std::move(subset));
  }

  return powerset;
}

}  // namespace hw1
