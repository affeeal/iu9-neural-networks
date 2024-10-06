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

class Data final : public IData {
  std::shared_ptr<const Symbol> x_;
  std::shared_ptr<const Eigen::VectorXd> y_;

 public:
  const Eigen::VectorXd& GetX() const override { return x_->scan; }
  const Eigen::VectorXd& GetY() const override { return *y_; }
};

class DataSupplier final : public IDataSupplier {
  struct Parametrization final {
    static constexpr std::array kLabels{L'0', L'1', L'2', L'3', L'4',
                                        L'5', L'6', L'7', L'8', L'9'};
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
    static constexpr double kTrainingRatio = 0.6;
    static constexpr double kValidationRatio = 0.2;
    static constexpr double kTestingRatio = 0.2;

    std::default_random_engine generator;

    Parametrization();
  };

  static const Parametrization& kParametrization;

  std::vector<std::shared_ptr<const IData>> training_;
  std::vector<std::shared_ptr<const IData>> validation_;
  std::vector<std::shared_ptr<const IData>> testing_;

 public:
  DataSupplier(const double low_score, const double high_score) {
    auto label_to_y =
        std::unordered_map<wchar_t, std::shared_ptr<const Eigen::VectorXd>>{};
    label_to_y.reserve(Parametrization::kLabelsNumber);
    for (std::size_t i = 0; i < Parametrization::kLabelsNumber; ++i) {
      auto y = std::make_shared<const Eigen::VectorXd>(
          Eigen::VectorXd::Zero(Parametrization::kLabelsNumber));
      label_to_y.insert({Parametrization::kLabels[i], std::move(y)});
    }

    // TODO: Remove duplication
    training_.reserve(kParametrization.training.size());
    for (auto&& symbol : kParametrization.training) {
      training_.push_back(
          std::make_shared<const Data>(symbol, label_to_y.at(symbol->label)));
    }

    validation_.reserve(kParametrization.validation.size());
    for (auto&& symbol : kParametrization.validation) {
      validation_.push_back(
          std::make_shared<const Data>(symbol, label_to_y.at(symbol->label)));
    }

    testing_.reserve(kParametrization.testing.size());
    for (auto&& symbol : kParametrization.testing) {
      testing_.push_back(
          std::make_shared<const Data>(symbol, label_to_y.at(symbol->label)));
    }
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
};

}  // namespace lab1
