#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "perceptron.h"

namespace lab1 {

constexpr std::size_t kScanSize = 20;
constexpr std::size_t kClassesCount = 20;

struct Symbol final {
  Eigen::VectorXd scan;  // развёртка
  std::string label;
};

class Data final : public nn::IData {
  std::shared_ptr<const Symbol> x_;
  std::shared_ptr<const Eigen::VectorXd> y_;

 public:
  Data(std::shared_ptr<const Symbol> x,
       std::shared_ptr<const Eigen::VectorXd> y)
      : x_(std::move(x)), y_(std::move(y)) {}

  const Eigen::VectorXd& GetX() const override { return x_->scan; }
  const Eigen::VectorXd& GetY() const override { return *y_; }
  std::string_view ToString() const override { return x_->label; }
};

class DataSupplier final : public nn::IDataSupplier {
  struct Parametrization;

  static const Parametrization& kParametrization;

  std::vector<std::shared_ptr<const nn::IData>> training_;
  std::vector<std::shared_ptr<const nn::IData>> validation_;
  std::vector<std::shared_ptr<const nn::IData>> testing_;

 public:
  DataSupplier(const double low_score, const double high_score);

  std::vector<std::shared_ptr<const nn::IData>> GetTrainingData() const override {
    return training_;
  }

  std::vector<std::shared_ptr<const nn::IData>> GetValidationData() const override {
    return validation_;
  }

  std::vector<std::shared_ptr<const nn::IData>> GetTestingData() const override {
    return testing_;
  }
};

}  // namespace lab1
