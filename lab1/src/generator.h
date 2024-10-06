#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <eigen3/Eigen/Core>
#include <iterator>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "data_supplier.h"
#include "perceptron.h"
#include "util.h"

namespace lab1 {

struct Symbol final {
  Eigen::VectorXd scan;  // развёртка
  wchar_t label;
};

constexpr std::size_t kSymbolScanSize = 20;
constexpr std::size_t kSymbolsCount = 10;  // TODO: Increase to 20

using InputVector = Eigen::Vector<double, kSymbolScanSize>;
using OutputVector = Eigen::Vector<double, kSymbolsCount>;

class Data final : public IData {
  Eigen::VectorXd input_;
  OutputVector output_;
  std::shared_ptr<const Eigen::VectorXd> y_;

 public:
  const Eigen::VectorXd& GetX() const override { return input_; }
  const Eigen::VectorXd& GetY() const override { return output_; }
};

class DataSupplier final : public IDataSupplier {
  struct Parametrization final {
    static constexpr double kTrainingRatio = 0.6;
    static constexpr double kValidationRatio = 0.2;
    static constexpr double kTestingRatio = 0.2;

    static_assert(kTrainingRatio + kValidationRatio + kTestingRatio == 1, "");

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

  static const Parametrization& kParametrization =
      Parametrization::GetInstance();

  std::vector<std::shared_ptr<const IData>> training_;
  std::vector<std::shared_ptr<const IData>> validation_;
  std::vector<std::shared_ptr<const IData>> testing_;

 public:
  DataSupplier(const double low_score, const double high_score) {}

  std::vector<std::shared_ptr<const IData>> GetTrainingData() const override {
    return training_;
  }

  std::vector<std::shared_ptr<const IData>> GetValidationData() const override {
    return validation_;
  }

  std::vector<std::shared_ptr<const IData>> GetTestingData() const override {
    return testing_;
  }
};

}  // namespace lab1
