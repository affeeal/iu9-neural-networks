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
  // requires std::input_iterator<Iter>
  void UpdateMiniBatch(const Iter mini_batch_begin, const Iter mini_batch_end,
                       const std::size_t mini_batch_size, const double eta) {
    auto nabla_weights = std::vector<Eigen::MatrixXd>{};
    nabla_weights.reserve(connections_number_);
    for (auto&& w : weights_) {
      // std::cout << w << std::endl;
      nabla_weights.push_back(Eigen::MatrixXd::Zero(w.rows(), w.cols()));
    }
    // std::cout << nabla_weights.back() << std::endl;

    auto nabla_biases = std::vector<Eigen::VectorXd>{};
    nabla_biases.reserve(connections_number_);
    for (auto&& b : biases_) {
      nabla_biases.push_back(Eigen::VectorXd::Zero(b.size()));
    }
    // std::cout << nabla_biases.back();

    for (auto it = mini_batch_begin; it != mini_batch_end; ++it) {
      const auto& data = **it;
      const auto [nabla_weights_part, nabla_biases_part] =
          Backpropagation(data.GetX(), data.GetY());
      // std::cout << '?' << '\n';
      for (std::size_t i = 0; i < connections_number_; ++i) {
        nabla_weights[i] += nabla_weights_part[i];
        nabla_biases[i] += nabla_biases_part[i];
      }
    }

    const std::size_t learning_rate = eta / mini_batch_size;
    for (std::size_t i = 0; i < connections_number_; ++i) {
      weights_[i] -= learning_rate * nabla_weights[i];
      biases_[i] -= learning_rate * nabla_biases[i];
    }
  }

  std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
  Backpropagation(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    const auto [zs, activations] = FeedforwardStage(x);
    // std::cout << '?' << '\n';
    assert(zs.size() == connections_number_);
    assert(activations.size() == layers_number_);

    auto delta = static_cast<Eigen::VectorXd>(
        cost_function_->PrimeActivations(y, activations.back()).array() *
        activation_functions_.back()->Prime(zs.back()).array());

    auto nabla_weights_reversed = std::vector<Eigen::MatrixXd>{};
    nabla_weights_reversed.reserve(connections_number_);
    nabla_weights_reversed.push_back(
        delta * std::prev(activations.cend(), 2)->transpose());

    auto nabla_biases_reversed = std::vector<Eigen::VectorXd>{};
    nabla_biases_reversed.reserve(connections_number_);
    nabla_biases_reversed.push_back(delta);

    for (int i = connections_number_ - 2; i > 0; --i) {
      // TODO: Move delta somehow?
      delta = static_cast<Eigen::VectorXd>(
          (weights_[i + 1].transpose() * delta).array() *
          activation_functions_[i]->Prime(zs[i]).array());
      nabla_weights_reversed.push_back(delta * activations[i].transpose());
      nabla_biases_reversed.push_back(delta);
    }

    return {{std::make_move_iterator(nabla_weights_reversed.rbegin()),
             std::make_move_iterator(nabla_weights_reversed.rend())},
            {std::make_move_iterator(nabla_biases_reversed.rbegin()),
             std::make_move_iterator(nabla_biases_reversed.rend())}};
  }

  std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  FeedforwardStage(const Eigen::VectorXd& x) {
    // std::cout << "?\n";
    // std::cout << x << '\n';
    std::vector<Eigen::VectorXd> zs, activations;
    zs.reserve(connections_number_);
    activations.reserve(layers_number_);

    auto activation = x;
    for (std::size_t i = 0; i < connections_number_; ++i) {
      auto z =
          static_cast<Eigen::VectorXd>(weights_[i] * activation + biases_[i]);
      activations.push_back(std::move(activation));
      // std::cout << &activation_functions_[i] << '\n';
      activation = activation_functions_[i]->Apply(z);
      // std::cout << "?\n";
      zs.push_back(std::move(z));
    }
    activations.push_back(std::move(activation));

    return {zs, activations};
  }

  std::size_t Evaluate(
      const std::vector<std::shared_ptr<const IData>>& testing);
};

}  // namespace lab1
