#pragma once

#include <memory>
#include <vector>

// clang-format off
#include <Eigen/Dense>
// clang-format on

namespace lab1 {

class IData {
 public:
  virtual ~IData() = default;

 public:
  virtual const Eigen::VectorXd& GetX() const = 0;
  virtual const Eigen::VectorXd& GetY() const = 0;
};

class IDataSupplier {
 public:
  virtual ~IDataSupplier() = default;

 public:
  virtual std::vector<std::shared_ptr<const IData>> GetTrainingData() const = 0;
  virtual std::vector<std::shared_ptr<const IData>> GetValidationData()
      const = 0;
  virtual std::vector<std::shared_ptr<const IData>> GetTestingData() const = 0;
};

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
  struct Parametrization;

  static const Parametrization& kParametrization;

  std::vector<std::shared_ptr<const IData>> training_;
  std::vector<std::shared_ptr<const IData>> validation_;
  std::vector<std::shared_ptr<const IData>> testing_;

 public:
  DataSupplier(const double low_score, const double high_score);

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
