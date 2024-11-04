#pragma once

#include <iostream>
#include <vector>

namespace nn {

template <typename T>
std::vector<std::vector<T>> GeneratePowerset(const std::vector<T>& set) {
  const auto set_size = set.size();
  const auto powerset_size = static_cast<std::size_t>(1 << set_size);

  auto powerset = std::vector<std::vector<T>>{};
  powerset.reserve(powerset_size);

  for (std::size_t i = 0; i < powerset_size; ++i) {
    auto subset = std::vector<T>{};

    for (std::size_t j = 0; j < set_size; ++j) {
      if (i & (1 << j)) {
        subset.push_back(set.at(j));
      }
    }

    powerset.push_back(std::move(subset));
  }

  return powerset;
}

}  // namespace nn
