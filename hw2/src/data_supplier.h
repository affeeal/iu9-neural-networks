#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "perceptron.h"

namespace hw2 {

constexpr std::size_t kScanSize = 784;
constexpr std::size_t kDigitsNumber = 10;

struct Data final : nn::IData {
  Eigen::VectorXd x, y;
  std::string label;

  const Eigen::VectorXd &GetX() const override { return x; }
  const Eigen::VectorXd &GetY() const override { return y; }
  std::string_view ToString() const override { return label; }
};

class DataSupplier final : public nn::IDataSupplier {
  std::vector<std::shared_ptr<const nn::IData>> training_, testing_,
      validation_;

 public:
  DataSupplier(const std::string &train_path, const std::string &test_path,
               const double false_score, const double true_score);

  std::vector<std::shared_ptr<const nn::IData>> GetTrainingData()
      const override {
    return training_;
  }
  std::vector<std::shared_ptr<const nn::IData>> GetValidationData()
      const override {
    return validation_;
  }
  std::vector<std::shared_ptr<const nn::IData>> GetTestingData()
      const override {
    return testing_;
  }
};

}  // namespace hw2
