#pragma once

#include <Eigen/Dense>
#include <memory>
#include <random>

#include "activation_function.h"
#include "cost_function.h"

namespace nn {

class IData {
 public:
  virtual ~IData() = default;

 public:
  virtual const Eigen::VectorXd &GetX() const = 0;
  virtual const Eigen::VectorXd &GetY() const = 0;
  virtual std::string_view ToString() const = 0;
};

class IDataSupplier {
 public:
  virtual ~IDataSupplier() = default;

 public:
  virtual std::size_t GetInputLayerSize() const = 0;
  virtual std::size_t GetOutputLayerSize() const = 0;

  virtual std::vector<std::shared_ptr<const IData>> GetTrainData() const = 0;
  virtual std::vector<std::shared_ptr<const IData>> GetTestData() const = 0;
  virtual std::vector<std::shared_ptr<const IData>> GetValidationData()
      const = 0;
};

struct SgdConfiguration final {
  std::size_t epochs;
  std::size_t mini_batch_size;
  double learning_rate;
  bool monitor_train_cost;
  bool monitor_train_accuracy;
  bool monitor_test_cost;
  bool monitor_test_accuracy;
};

struct Metric final {
  std::vector<double> train_cost, train_accuracy;
  std::vector<double> test_cost, test_accuracy;
};

class Perceptron final {
  std::random_device device_;
  std::default_random_engine generator_;
  std::unique_ptr<ICostFunction> cost_function_;
  std::size_t layers_number_, connections_number_;
  std::vector<Eigen::MatrixXd> weights_;
  std::vector<Eigen::VectorXd> biases_;
  std::vector<std::unique_ptr<IActivationFunction>> activation_functions_;

 public:
  Perceptron(
      std::unique_ptr<ICostFunction> &&cost_function,
      std::vector<std::unique_ptr<IActivationFunction>> &&activation_functions,
      const std::vector<std::size_t> &layers_sizes);

  Eigen::VectorXd Feedforward(const Eigen::VectorXd &x) const;

  Metric Sgd(const std::vector<std::shared_ptr<const IData>> &train,
             const std::vector<std::shared_ptr<const IData>> &test,
             const SgdConfiguration &cfg);

  Metric SgdNag(const std::vector<std::shared_ptr<const IData>> &train,
                const std::vector<std::shared_ptr<const IData>> &test,
                const SgdConfiguration &cfg, const double gamma);

  Metric SgdAdagrad(const std::vector<std::shared_ptr<const IData>> &train,
                    const std::vector<std::shared_ptr<const IData>> &test,
                    const SgdConfiguration &cfg, const double epsilon);

  Metric SgdAdam(const std::vector<std::shared_ptr<const IData>> &train,
                 const std::vector<std::shared_ptr<const IData>> &test,
                 const SgdConfiguration &cfg, const double beta1,
                 const double beta2, const double epsilon);

 private:
  // TODO: Use concepts
  template <typename Iter>
  void UpdateSgd(const Iter mini_batch_begin, const Iter mini_batch_end,
                 const std::size_t mini_batch_size, const double learning_rate);

  // EMA means Exponential Moving Average
  template <typename Iter>
  void UpdateSgdNag(std::vector<Eigen::MatrixXd> &delta_weights_ema,
                    std::vector<Eigen::VectorXd> &delta_biases_ema,
                    const Iter mini_batch_begin, const Iter mini_batch_end,
                    const std::size_t mini_batch_size,
                    const double learning_rate, const double gamma);

  template <typename Iter>
  void UpdateSgdAdagrad(
      std::vector<Eigen::MatrixXd> &weights_gradient_squares_sum,
      std::vector<Eigen::VectorXd> &biases_gradient_squares_sum,
      const Iter mini_batch_begin, const Iter mini_batch_end,
      const std::size_t mini_batch_size, const double learning_rate);

  template <typename Iter>
  void UpdateSgdAdam(std::vector<Eigen::MatrixXd> &weights_gradient_ema,
                     std::vector<Eigen::VectorXd> &biases_gradient_ema,
                     std::vector<Eigen::MatrixXd> &weights_squared_gradient_ema,
                     std::vector<Eigen::VectorXd> &biases_squared_gradient_ema,
                     const Iter mini_batch_begin, const Iter mini_batch_end,
                     const std::size_t mini_batch_size, const std::size_t epoch,
                     const double learning_rate, const double beta1,
                     const double beta2);

 private:
  struct Parameters {
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
  };

  Parameters CreateParameters(const double initial_value) const;

  template <typename Iter>
  Parameters GradientWrtParameters(const Iter mini_batch_begin,
                                   const Iter mini_batch_end,
                                   const std::size_t mini_batch_size) const;

  Parameters Backpropagation(const Eigen::VectorXd &x,
                             const Eigen::VectorXd &y) const;

  std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  FeedforwardDetailed(const Eigen::VectorXd &x) const;

  Metric CreateMetric(const SgdConfiguration &cfg) const;

  void WriteMetric(Metric &metric, const std::size_t epoch,
                   const std::vector<std::shared_ptr<const IData>> &train,
                   const std::vector<std::shared_ptr<const IData>> &test,
                   const SgdConfiguration &cfg) const;

  template <typename Iter>
  std::size_t CalculateAccuracy(const Iter begin, const Iter end) const;

  template <typename Iter>
  double CalculateCost(const Iter begin, const Iter end) const;
};

}  // namespace nn
