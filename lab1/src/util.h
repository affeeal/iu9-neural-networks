#pragma once

#include <vector>

namespace lab1 {

template <typename T>
std::vector<std::vector<T>> GetPowerset(const std::vector<T>& set) {
  const auto set_size = set.size();
  const auto powerset_size = 2 << set_size;

  auto powerset = std::vector<std::vector<T>>{};
  powerset.reserve(powerset_size);

  for (auto i = 0; i < powerset_size; ++i) {
    auto subset = std::vector<T>{};

    for (auto j = 0; j < set_size; ++j) {
      if (i & (1 << j)) {
        subset.push_back(set.at(j));
      }
    }

    powerset.push_back(std::move(subset));
  }

  return powerset;
}

}  // namespace lab1
