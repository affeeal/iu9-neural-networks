#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#define CHECK(condition)                                                    \
  do {                                                                      \
    if (!(condition))                                                       \
      throw std::runtime_error(std::string(__FILE__) + ":" +                \
                               std::to_string(__LINE__) + ": " #condition); \
  } while (false)

inline void Near(double actual, double expected, double tolerance = 1e-9) {
  CHECK(std::isfinite(actual));
  CHECK(std::abs(actual - expected) <= tolerance);
}

template <typename F>
void Throws(F&& action) {
  bool thrown = false;
  try {
    action();
  } catch (const std::exception&) {
    thrown = true;
  }
  CHECK(thrown);
}
