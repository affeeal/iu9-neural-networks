#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

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

}  // namespace lab1
