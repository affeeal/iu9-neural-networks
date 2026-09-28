#include "perceptron.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include "output_layer.h"

namespace nn {

Perceptron::Perceptron(
    std::unique_ptr<ICostFunction> &&cost_function,
    std::vector<std::unique_ptr<IActivationFunction>> &&activation_functions,
    const std::vector<std::size_t> &layers_sizes, std::uint32_t seed)
    : generator_(seed),
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

  if (!cost_function_ ||
      std::any_of(layers_sizes.begin(), layers_sizes.end(),
                  [](auto size) { return size == 0; }) ||
      std::any_of(activation_functions_.begin(), activation_functions_.end(),
                  [](const auto &activation) { return !activation; })) {
    throw std::invalid_argument(
        "Layers must be nonempty; functions must not be null");
  }
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  const auto random_value = [&] { return distribution(generator_); };
  weights_.reserve(connections_number_);
  biases_.reserve(connections_number_);
  for (std::size_t i = 0; i < connections_number_; ++i) {
    weights_.push_back(Eigen::MatrixXd::NullaryExpr(
        layers_sizes[i + 1], layers_sizes[i], random_value));
    biases_.push_back(
        Eigen::VectorXd::NullaryExpr(layers_sizes[i + 1], random_value));
  }
}

Eigen::VectorXd Perceptron::Feedforward(const Eigen::VectorXd &x) const {
  if (x.size() != weights_.front().cols() || !x.allFinite()) {
    throw std::invalid_argument("Invalid input vector");
  }
  auto activation = x;
  for (std::size_t i = 0; i < connections_number_; ++i) {
    activation =
        activation_functions_[i]->Apply(weights_[i] * activation + biases_[i]);
  }
  return activation;
}

Metric Perceptron::StochasticGradientSearch(
    const std::vector<std::shared_ptr<const IData>> &training,
    const std::vector<std::shared_ptr<const IData>> &testing,
    const Config &cfg) {
  ValidateTraining(training, testing, cfg);
  const auto training_size = training.size();
  const auto whole_mini_batches_number = training_size / cfg.mini_batch_size;
  const auto remainder_mini_batch_size = training_size % cfg.mini_batch_size;

  auto training_shuffled = std::vector(training.begin(), training.end());
  auto metric = GetMetric(cfg);
  for (std::size_t i = 0; i < cfg.epochs; ++i) {
    std::shuffle(training_shuffled.begin(), training_shuffled.end(),
                 generator_);
    auto it = training_shuffled.begin();
    for (std::size_t i = 0; i < whole_mini_batches_number; ++i) {
      auto end = it + cfg.mini_batch_size;
      UpdateMiniBatch(it, end, cfg.mini_batch_size, cfg.eta);
      it = std::move(end);
    }
    if (remainder_mini_batch_size != 0) {
      UpdateMiniBatch(it, it + remainder_mini_batch_size,
                      remainder_mini_batch_size, cfg.eta);
    }
    WriteMetric(metric, i, training, testing, cfg);
  }
  return metric;
}

template <typename Iter>
void Perceptron::UpdateMiniBatch(const Iter mini_batch_begin,
                                 const Iter mini_batch_end,
                                 const std::size_t mini_batch_size,
                                 const double eta) {
  auto nabla_weights = std::vector<Eigen::MatrixXd>{};
  nabla_weights.reserve(connections_number_);
  for (auto &&w : weights_) {
    nabla_weights.push_back(Eigen::MatrixXd::Zero(w.rows(), w.cols()));
  }

  auto nabla_biases = std::vector<Eigen::VectorXd>{};
  nabla_biases.reserve(connections_number_);
  for (auto &&b : biases_) {
    nabla_biases.push_back(Eigen::VectorXd::Zero(b.size()));
  }

  for (auto it = mini_batch_begin; it != mini_batch_end; ++it) {
    const auto &data = **it;
    const auto [nabla_weights_part, nabla_biases_part] =
        Backpropagation(data.GetX(), data.GetY());
    for (std::size_t i = 0; i < connections_number_; ++i) {
      nabla_weights[i] += nabla_weights_part[i];
      nabla_biases[i] += nabla_biases_part[i];
    }
  }

  const auto learning_rate = eta / mini_batch_size;
  for (std::size_t i = 0; i < connections_number_; ++i) {
    weights_[i] -= learning_rate * nabla_weights[i];
    biases_[i] -= learning_rate * nabla_biases[i];
  }
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
Perceptron::Backpropagation(const Eigen::VectorXd &x,
                            const Eigen::VectorXd &y) {
  const auto [zs, activations] = FeedforwardDetailed(x);
  assert(zs.size() == connections_number_);
  assert(activations.size() == layers_number_);

  auto delta = OutputDelta(*activation_functions_.back(), *cost_function_, y,
                           zs.back(), activations.back());

  auto nabla_weights_reversed = std::vector<Eigen::MatrixXd>{};
  nabla_weights_reversed.reserve(connections_number_);
  nabla_weights_reversed.push_back(
      delta * std::prev(activations.cend(), 2)->transpose());

  auto nabla_biases_reversed = std::vector<Eigen::VectorXd>{};
  nabla_biases_reversed.reserve(connections_number_);
  nabla_biases_reversed.push_back(delta);

  for (std::size_t i = connections_number_ - 1; i-- > 0;) {
    delta = (weights_[i + 1] * activation_functions_[i]->Jacobian(zs[i]))
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
Perceptron::FeedforwardDetailed(const Eigen::VectorXd &x) const {
  std::vector<Eigen::VectorXd> zs, activations;
  zs.reserve(connections_number_);
  activations.reserve(layers_number_);

  auto activation = x;
  for (std::size_t i = 0; i < connections_number_; ++i) {
    auto z =
        static_cast<Eigen::VectorXd>(weights_[i] * activation + biases_[i]);
    activations.push_back(std::move(activation));
    activation = activation_functions_[i]->Apply(z);
    zs.push_back(std::move(z));
  }
  activations.push_back(std::move(activation));

  return {zs, activations};
}

Metric Perceptron::GetMetric(const Config &param) const {
  auto metric = Metric{};
  if (param.monitor_training_cost) {
    metric.training_cost.reserve(param.epochs);
  }
  if (param.monitor_training_accuracy) {
    metric.training_accuracy.reserve(param.epochs);
  }
  if (param.monitor_testing_cost) {
    metric.testing_cost.reserve(param.epochs);
  }
  if (param.monitor_testing_accuracy) {
    metric.testing_accuracy.reserve(param.epochs);
  }
  return metric;
}

void Perceptron::WriteMetric(
    Metric &metric, const std::size_t epoch,
    const std::vector<std::shared_ptr<const IData>> &training,
    const std::vector<std::shared_ptr<const IData>> &testing,
    const Config &cfg) const {
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
    metric.testing_cost.push_back(testing_cost);
    oss << " testing cost: " << testing_cost << ";";
  }
  if (cfg.monitor_testing_accuracy) {
    const auto testing_accuracy = Accuracy(testing.begin(), testing.end());
    metric.testing_accuracy.push_back(testing_accuracy);
    oss << " testing accuracy: " << testing_accuracy << "/" << testing.size()
        << ";";
  }
  spdlog::info("{}", oss.str());
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
    const auto [logits, activations] = FeedforwardDetailed(instance.GetX());
    cost += OutputCost(*activation_functions_.back(), *cost_function_,
                       instance.GetY(), logits.back());
  }
  return cost / instances_count;
}

void Perceptron::ValidateTraining(
    const std::vector<std::shared_ptr<const IData>> &train,
    const std::vector<std::shared_ptr<const IData>> &test,
    const Config &cfg) const {
  if (train.empty() || cfg.epochs == 0 || cfg.mini_batch_size == 0 ||
      !std::isfinite(cfg.eta) || cfg.eta <= 0 ||
      ((cfg.monitor_testing_cost || cfg.monitor_testing_accuracy) &&
       test.empty())) {
    throw std::invalid_argument(
        "Invalid training configuration or empty dataset");
  }
  for (const auto *dataset : {&train, &test}) {
    for (const auto &item : *dataset) {
      if (!item || item->GetX().size() != weights_.front().cols() ||
          item->GetY().size() != weights_.back().rows() ||
          !item->GetX().allFinite() || !item->GetY().allFinite()) {
        throw std::invalid_argument("Invalid training sample");
      }
      if (IsCategorical(*cost_function_) &&
          ((item->GetY().array() < 0).any() || item->GetY().sum() <= 0)) {
        throw std::invalid_argument(
            "Categorical targets must be nonnegative with positive mass");
      }
    }
  }
}

}  // namespace nn
