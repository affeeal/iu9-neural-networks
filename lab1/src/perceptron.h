#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "activation_function.h"
#include "cost_function.h"
#include "data_supplier.h"

namespace lab1 {

class Perceptron final {
  std::default_random_engine generator_;
  std::unique_ptr<ICostFunction> cost_function_;
  std::size_t layers_number_, connections_number_;
  std::vector<Eigen::MatrixXd> weights_;
  std::vector<Eigen::VectorXd> biases_;
  std::vector<std::unique_ptr<IActivationFunction>> activation_functions_;

 public:
  Perceptron(
      std::unique_ptr<ICostFunction>&& cost_function,
      std::vector<std::unique_ptr<IActivationFunction>>&& activation_functions,
      const std::vector<std::size_t>& layers_sizes);

  Eigen::VectorXd Feedforward(const Eigen::VectorXd& x) const;

  void StochasticGradientSearch(
      const std::vector<std::shared_ptr<const IData>>& training,
      const std::size_t epochs, const std::size_t mini_batch_size,
      const double eta,
      const std::vector<std::shared_ptr<const IData>>& testing);

 private:
  template <typename Iter>
  void UpdateMiniBatch(const Iter mini_batch_begin, const Iter mini_batch_end,
                       const std::size_t mini_batch_size, const double eta);

  std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
  Backpropagation(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

  std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  FeedforwardPass(const Eigen::VectorXd& x);

  std::size_t Evaluate(
      const std::vector<std::shared_ptr<const IData>>& testing);
};

}  // namespace lab1
