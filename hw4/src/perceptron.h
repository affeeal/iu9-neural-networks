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
  virtual std::vector<std::shared_ptr<const IData>> GetTrainingData() const = 0;
  virtual std::vector<std::shared_ptr<const IData>> GetValidationData()
      const = 0;
  virtual std::vector<std::shared_ptr<const IData>> GetTestingData() const = 0;
};

struct SgdConfiguration final {
  std::size_t epochs;
  std::size_t mini_batch_size;
  double learning_rate;
  bool monitor_training_cost;
  bool monitor_training_accuracy;
  bool monitor_testing_cost;
  bool monitor_testing_accuracy;
};

struct Metric final {
  std::vector<double> training_cost, training_accuracy;
  std::vector<double> testing_cost, testing_accuracy;
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

  Metric Sgd(const std::vector<std::shared_ptr<const IData>> &training,
             const std::vector<std::shared_ptr<const IData>> &testing,
             const SgdConfiguration &cfg);

  Metric SgdNag(const std::vector<std::shared_ptr<const IData>> &training,
                const std::vector<std::shared_ptr<const IData>> &testing,
                const SgdConfiguration &cfg, const double momentum);

  Metric SgdAdagrad(const std::vector<std::shared_ptr<const IData>> &training,
                    const std::vector<std::shared_ptr<const IData>> &testing,
                    const SgdConfiguration &cfg, const double epsilon);

 private:
  template <typename Iter>  // TODO: Use concepts
  void UpdateSgd(const Iter mini_batch_begin, const Iter mini_batch_end,
                 const std::size_t mini_batch_size, const double learning_rate);

  template <typename Iter>
  void UpdateSgdNag(std::vector<Eigen::MatrixXd> &weights_momentum,
                    std::vector<Eigen::VectorXd> &biases_momentum,
                    const Iter mini_batch_begin, const Iter mini_batch_end,
                    const std::size_t mini_batch_size,
                    const double learning_rate, const double momentum);

  template <typename Iter>
  void UpdateSgdAdagrad(std::vector<Eigen::MatrixXd> &nabla_weights_squares,
                        std::vector<Eigen::VectorXd> &nabla_biases_squares,
                        const Iter mini_batch_begin, const Iter mini_batch_end,
                        const std::size_t mini_batch_size,
                        const double learning_rate);

  std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
  Backpropagation(const Eigen::VectorXd &x, const Eigen::VectorXd &y);

  std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  FeedforwardDetailed(const Eigen::VectorXd &x);

  std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
  CreateParameters(const double initial_value) const;

  Metric CreateMetric(const SgdConfiguration &cfg) const;

  void WriteMetric(Metric &metric, const std::size_t epoch,
                   const std::vector<std::shared_ptr<const IData>> &training,
                   const std::vector<std::shared_ptr<const IData>> &testing,
                   const SgdConfiguration &cfg) const;

  template <typename Iter>
  std::size_t Accuracy(const Iter begin, const Iter end) const;

  template <typename Iter>
  double Cost(const Iter begin, const Iter end) const;
};

}  // namespace nn
