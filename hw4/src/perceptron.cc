#include "perceptron.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <sstream>

namespace nn {

Perceptron::Perceptron(
    std::unique_ptr<ICostFunction> &&cost_function,
    std::vector<std::unique_ptr<IActivationFunction>> &&activation_functions,
    const std::vector<std::size_t> &layers_sizes)
    : generator_(device_()),
      cost_function_(std::move(cost_function)),
      layers_number_(layers_sizes.size()),
      connections_number_(layers_number_ - 1),
      activation_functions_(std::move(activation_functions)) {
  if (layers_number_ < 2) {
    throw std::runtime_error("Perceptron must have at least two layers");
  }

  if (activation_functions_.size() != connections_number_) {
    throw std::runtime_error(
        "Activation functions number must be equal to layers number minus one");
  }

  weights_.reserve(connections_number_);
  biases_.reserve(connections_number_);
  for (std::size_t i = 0; i < connections_number_; ++i) {
    weights_.push_back(
        Eigen::MatrixXd::Random(layers_sizes[i + 1], layers_sizes[i]));
    biases_.push_back(Eigen::VectorXd::Random(layers_sizes[i + 1]));
  }
}

Eigen::VectorXd Perceptron::Feedforward(const Eigen::VectorXd &x) const {
  auto activation = x;
  for (std::size_t i = 0; i < connections_number_; ++i) {
    activation =
        activation_functions_[i]->Apply(weights_[i] * activation + biases_[i]);
  }
  return activation;
}

Metric Perceptron::Sgd(
    const std::vector<std::shared_ptr<const IData>> &training,
    const std::vector<std::shared_ptr<const IData>> &testing,
    const SgdConfiguration &cfg) {
  const auto training_size = training.size();
  const auto whole_mini_batches_number = training_size / cfg.mini_batch_size;
  const auto remainder_mini_batch_size = training_size % cfg.mini_batch_size;

  auto training_shuffled = std::vector(training.begin(), training.end());
  auto metric = CreateMetric(cfg);
  for (std::size_t i = 0; i < cfg.epochs; ++i) {
    std::shuffle(training_shuffled.begin(), training_shuffled.end(),
                 generator_);
    auto it = training_shuffled.begin();
    for (std::size_t i = 0; i < whole_mini_batches_number; ++i) {
      auto end = it + cfg.mini_batch_size;
      UpdateSgd(it, end, cfg.mini_batch_size, cfg.learning_rate);
      it = std::move(end);
    }
    if (remainder_mini_batch_size != 0) {
      UpdateSgd(it, it + remainder_mini_batch_size, remainder_mini_batch_size,
                cfg.learning_rate);
    }
    WriteMetric(metric, i, training, testing, cfg);
  }
  return metric;
}

Metric Perceptron::SgdNag(
    const std::vector<std::shared_ptr<const IData>> &training,
    const std::vector<std::shared_ptr<const IData>> &testing,
    const SgdConfiguration &cfg, const double momentum) {
  if (momentum < 0 || momentum > 1) {
    throw std::runtime_error("The momentum must belong to [0, 1]");
  }

  const auto training_size = training.size();
  const auto whole_mini_batches_number = training_size / cfg.mini_batch_size;
  const auto remainder_mini_batch_size = training_size % cfg.mini_batch_size;

  auto [weights_momentum, biases_momentum] = CreateParameters(0);
  auto training_shuffled = std::vector(training.begin(), training.end());
  auto metric = CreateMetric(cfg);
  for (std::size_t i = 0; i < cfg.epochs; ++i) {
    std::shuffle(training_shuffled.begin(), training_shuffled.end(),
                 generator_);
    auto it = training_shuffled.begin();
    for (std::size_t i = 0; i < whole_mini_batches_number; ++i) {
      auto end = it + cfg.mini_batch_size;
      UpdateSgdNag(weights_momentum, biases_momentum, it, end,
                   cfg.mini_batch_size, cfg.learning_rate, momentum);
      it = std::move(end);
    }
    if (remainder_mini_batch_size != 0) {
      UpdateSgdNag(weights_momentum, biases_momentum, it,
                   it + remainder_mini_batch_size, remainder_mini_batch_size,
                   cfg.learning_rate, momentum);
    }
    WriteMetric(metric, i, training, testing, cfg);
  }
  return metric;
}

Metric Perceptron::SgdAdagrad(
    const std::vector<std::shared_ptr<const IData>> &training,
    const std::vector<std::shared_ptr<const IData>> &testing,
    const SgdConfiguration &cfg, const double epsilon) {
  if (epsilon <= 0) {
    throw std::runtime_error("The epsilon must be strictly greater than 0");
  }

  const auto training_size = training.size();
  const auto whole_mini_batches_number = training_size / cfg.mini_batch_size;
  const auto remainder_mini_batch_size = training_size % cfg.mini_batch_size;

  auto [nabla_weights_squares, nabla_biases_square] = CreateParameters(epsilon);
  auto training_shuffled = std::vector(training.begin(), training.end());
  auto metric = CreateMetric(cfg);
  for (std::size_t i = 0; i < cfg.epochs; ++i) {
    std::shuffle(training_shuffled.begin(), training_shuffled.end(),
                 generator_);
    auto it = training_shuffled.begin();
    for (std::size_t i = 0; i < whole_mini_batches_number; ++i) {
      auto end = it + cfg.mini_batch_size;
      UpdateSgdAdagrad(nabla_weights_squares, nabla_biases_square, it, end,
                       cfg.mini_batch_size, cfg.learning_rate);
      it = std::move(end);
    }
    if (remainder_mini_batch_size != 0) {
      UpdateSgdAdagrad(nabla_weights_squares, nabla_biases_square, it,
                       it + remainder_mini_batch_size,
                       remainder_mini_batch_size, cfg.learning_rate);
    }
    WriteMetric(metric, i, training, testing, cfg);
  }

  return metric;
}

template <typename Iter>
void Perceptron::UpdateSgd(const Iter mini_batch_begin,
                           const Iter mini_batch_end,
                           const std::size_t mini_batch_size,
                           const double learning_rate) {
  auto [nabla_weights, nabla_biases] = CreateParameters(0);
  for (auto it = mini_batch_begin; it != mini_batch_end; ++it) {
    const auto &data = **it;
    const auto [nabla_weights_contribution, nabla_biases_contribution] =
        Backpropagation(data.GetX(), data.GetY());
    for (std::size_t i = 0; i < connections_number_; ++i) {
      nabla_weights[i] += nabla_weights_contribution[i];
      nabla_biases[i] += nabla_biases_contribution[i];
    }
  }

  const auto factor = learning_rate / mini_batch_size;
  for (std::size_t i = 0; i < connections_number_; ++i) {
    weights_[i] -= factor * nabla_weights[i];
    biases_[i] -= factor * nabla_biases[i];
  }
}

template <typename Iter>
void Perceptron::UpdateSgdNag(std::vector<Eigen::MatrixXd> &weights_momentum,
                              std::vector<Eigen::VectorXd> &biases_momentum,
                              const Iter mini_batch_begin,
                              const Iter mini_batch_end,
                              const std::size_t mini_batch_size,
                              const double learning_rate,
                              const double momentum) {
  for (std::size_t i = 0; i < connections_number_; ++i) {
    weights_[i] -= momentum * weights_momentum[i];
    biases_[i] -= momentum * biases_momentum[i];
  }

  auto [nabla_weights, nabla_biases] = CreateParameters(0);
  for (auto it = mini_batch_begin; it != mini_batch_end; ++it) {
    const auto &data = **it;
    const auto [nabla_weights_contribution, nabla_biases_contribution] =
        Backpropagation(data.GetX(), data.GetY());
    for (std::size_t i = 0; i < connections_number_; ++i) {
      nabla_weights[i] += nabla_weights_contribution[i];
      nabla_biases[i] += nabla_biases_contribution[i];
    }
  }

  const auto factor = learning_rate / mini_batch_size;
  for (std::size_t i = 0; i < connections_number_; ++i) {
    const auto saved_weights_momentum = momentum * weights_momentum[i];
    weights_momentum[i] = saved_weights_momentum + factor * nabla_weights[i];
    // Addition of ths saved_weights_momentum is a compensation of the
    // subtraction above.
    weights_[i] += saved_weights_momentum - weights_momentum[i];

    const auto saved_biases_momentum = momentum * biases_momentum[i];
    biases_momentum[i] = saved_biases_momentum + factor * nabla_biases[i];
    biases_[i] += saved_biases_momentum - biases_momentum[i];
  }
}

template <typename Iter>
void Perceptron::UpdateSgdAdagrad(
    std::vector<Eigen::MatrixXd> &nabla_weights_squares,
    std::vector<Eigen::VectorXd> &nabla_biases_squares,
    const Iter mini_batch_begin, const Iter mini_batch_end,
    const std::size_t mini_batch_size, const double learning_rate) {
  auto [nabla_weights, nabla_biases] = CreateParameters(0);
  for (auto it = mini_batch_begin; it != mini_batch_end; ++it) {
    const auto &data = **it;
    const auto [nabla_weights_part, nabla_biases_part] =
        Backpropagation(data.GetX(), data.GetY());
    for (std::size_t i = 0; i < connections_number_; ++i) {
      nabla_weights[i] += nabla_weights_part[i];
      nabla_biases[i] += nabla_biases_part[i];
    }
  }

  for (std::size_t i = 0; i < connections_number_; ++i) {
    const auto weights_gradient = nabla_weights[i] / mini_batch_size;
    nabla_weights_squares[i] += weights_gradient.array().pow(2).matrix();
    weights_[i] -=
        learning_rate / nabla_weights_squares[i].lpNorm<2>() * weights_gradient;

    const auto biases_gradient = nabla_biases[i] / mini_batch_size;
    nabla_biases_squares[i] += biases_gradient.array().pow(2).matrix();
    biases_[i] -=
        learning_rate / nabla_biases_squares[i].lpNorm<2>() * biases_gradient;
  }
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
Perceptron::Backpropagation(const Eigen::VectorXd &x,
                            const Eigen::VectorXd &y) {
  const auto [linear_values, activations] = FeedforwardDetailed(x);
  assert(linear_values.size() == connections_number_);
  assert(activations.size() == layers_number_);

  auto delta = static_cast<Eigen::VectorXd>(
      activation_functions_.back()->Jacobian(linear_values.back()).transpose() *
      cost_function_->GradientWrtActivations(y, activations.back()));

  auto nabla_weights_reversed = std::vector<Eigen::MatrixXd>{};
  nabla_weights_reversed.reserve(connections_number_);
  nabla_weights_reversed.push_back(
      delta * std::prev(activations.cend(), 2)->transpose());

  auto nabla_biases_reversed = std::vector<Eigen::VectorXd>{};
  nabla_biases_reversed.reserve(connections_number_);
  nabla_biases_reversed.push_back(delta);

  for (int i = connections_number_ - 2; i >= 0; --i) {
    delta =
        (weights_[i + 1] * activation_functions_[i]->Jacobian(linear_values[i]))
            .transpose() *
        delta;
    nabla_weights_reversed.push_back(delta * activations[i].transpose());
    nabla_biases_reversed.push_back(delta);
  }

  return {{std::make_move_iterator(nabla_weights_reversed.rbegin()),
           std::make_move_iterator(nabla_weights_reversed.rend())},
          {std::make_move_iterator(nabla_biases_reversed.rbegin()),
           std::make_move_iterator(nabla_biases_reversed.rend())}};
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
Perceptron::FeedforwardDetailed(const Eigen::VectorXd &x) {
  std::vector<Eigen::VectorXd> linear_values, activations;
  linear_values.reserve(connections_number_);
  activations.reserve(layers_number_);

  auto activation = x;
  for (std::size_t i = 0; i < connections_number_; ++i) {
    auto linear_value =
        static_cast<Eigen::VectorXd>(weights_[i] * activation + biases_[i]);
    activations.push_back(std::move(activation));
    activation = activation_functions_[i]->Apply(linear_value);
    linear_values.push_back(std::move(linear_value));
  }
  activations.push_back(std::move(activation));

  return {linear_values, activations};
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
Perceptron::CreateParameters(const double initial_value) const {
  auto weights = std::vector<Eigen::MatrixXd>{};
  weights.reserve(weights_.size());
  for (auto &&w : weights_) {
    auto m = Eigen::MatrixXd(w.rows(), w.cols());
    m.setConstant(initial_value);
    weights.push_back(std::move(m));
  }

  auto biases = std::vector<Eigen::VectorXd>{};
  biases.reserve(biases_.size());
  for (auto &&b : biases_) {
    auto v = Eigen::VectorXd(b.size());
    v.setConstant(initial_value);
    biases.push_back(std::move(v));
  }

  return {weights, biases};
}

Metric Perceptron::CreateMetric(const SgdConfiguration &cfg) const {
  auto metric = Metric{};
  if (cfg.monitor_training_cost) {
    metric.training_cost.reserve(cfg.epochs);
  }
  if (cfg.monitor_training_accuracy) {
    metric.training_accuracy.reserve(cfg.epochs);
  }
  if (cfg.monitor_testing_cost) {
    metric.testing_cost.reserve(cfg.epochs);
  }
  if (cfg.monitor_testing_accuracy) {
    metric.testing_accuracy.reserve(cfg.epochs);
  }
  return metric;
}

void Perceptron::WriteMetric(
    Metric &metric, const std::size_t epoch,
    const std::vector<std::shared_ptr<const IData>> &training,
    const std::vector<std::shared_ptr<const IData>> &testing,
    const SgdConfiguration &cfg) const {
  std::stringstream oss;
  oss << "Epoch " << epoch << ";";
  if (cfg.monitor_training_cost) {
    const auto training_cost = Cost(training.begin(), training.end());
    metric.training_cost.push_back(training_cost);
    oss << " training cost: " << training_cost << ";";
  }
  if (cfg.monitor_training_accuracy) {
    const auto training_accuracy = Accuracy(training.begin(), training.end());
    metric.training_accuracy.push_back(training_accuracy);
    oss << " training accuracy: " << training_accuracy << "/" << training.size()
        << ";";
  }
  if (cfg.monitor_testing_cost) {
    const auto testing_cost = Cost(testing.begin(), testing.end());
    metric.testing_cost.push_back(Cost(testing.begin(), testing.end()));
    oss << " testing cost: " << testing_cost << ";";
  }
  if (cfg.monitor_testing_accuracy) {
    const auto testing_accuracy = Accuracy(testing.begin(), testing.end());
    metric.testing_accuracy.push_back(testing_accuracy);
    oss << " testing accuracy: " << testing_accuracy << "/" << testing.size()
        << ";";
  }
  spdlog::info(oss.str());
}

template <typename Iter>
std::size_t Perceptron::Accuracy(const Iter begin, const Iter end) const {
  std::size_t right_predictions = 0;
  for (auto it = begin; it != end; ++it) {
    const IData &instance = **it;
    Eigen::Index max_activation_expected, max_activation_actual;
    instance.GetY().maxCoeff(&max_activation_expected);
    Feedforward(instance.GetX()).maxCoeff(&max_activation_actual);
    if (max_activation_expected == max_activation_actual) {
      ++right_predictions;
    }
  }
  return right_predictions;
}

template <typename Iter>
double Perceptron::Cost(const Iter begin, const Iter end) const {
  double cost = 0;
  std::size_t instances_count = 0;
  for (auto it = begin; it != end; ++it, ++instances_count) {
    const IData &instance = **it;
    const auto activation = Feedforward(instance.GetX());
    cost += cost_function_->Apply(instance.GetY(), activation);
  }
  return cost / instances_count;
}

}  // namespace nn
