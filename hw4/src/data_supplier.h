#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "perceptron.h"

namespace nn {

struct Data final : nn::IData {
  Eigen::VectorXd x, y;
  std::string label;

  const Eigen::VectorXd &GetX() const override { return x; }
  const Eigen::VectorXd &GetY() const override { return y; }
  std::string_view ToString() const override { return label; }
};

class DataSupplier final : public nn::IDataSupplier {
  std::vector<std::shared_ptr<const nn::IData>> train_, test_, validation_;

 public:
  DataSupplier(const std::string &train_path, const std::string &test_path,
               const double false_score, const double true_score);

  std::size_t GetInputLayerSize() const override;
  std::size_t GetOutputLayerSize() const override;

  std::vector<std::shared_ptr<const nn::IData>> GetTrainData() const override;
  std::vector<std::shared_ptr<const nn::IData>> GetValidationData()
      const override;
  std::vector<std::shared_ptr<const nn::IData>> GetTestData() const override;
};

}  // namespace nn
