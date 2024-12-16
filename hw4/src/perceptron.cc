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

Metric Perceptron::Sgd(const std::vector<std::shared_ptr<const IData>> &train,
                       const std::vector<std::shared_ptr<const IData>> &test,
                       const SgdConfiguration &cfg) {
  const auto train_size = train.size();
  const auto whole_mini_batches_number = train_size / cfg.mini_batch_size;
  const auto remainder_mini_batch_size = train_size % cfg.mini_batch_size;

  auto train_shuffled = std::vector(train.begin(), train.end());
  auto metric = CreateMetric(cfg);
  for (std::size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
    std::shuffle(train_shuffled.begin(), train_shuffled.end(), generator_);
    auto it = train_shuffled.begin();
    for (std::size_t i = 0; i < whole_mini_batches_number; ++i) {
      auto end = it + cfg.mini_batch_size;
      UpdateSgd(it, end, cfg.mini_batch_size, cfg.learning_rate);
      it = std::move(end);
    }

    if (remainder_mini_batch_size != 0) {
      UpdateSgd(it, it + remainder_mini_batch_size, remainder_mini_batch_size,
                cfg.learning_rate);
    }
    WriteMetric(metric, epoch, train, test, cfg);
  }

  return metric;
}

Metric Perceptron::SgdNag(
    const std::vector<std::shared_ptr<const IData>> &train,
    const std::vector<std::shared_ptr<const IData>> &test,
    const SgdConfiguration &cfg, const double gamma) {
  if (gamma < 0 || gamma > 1) {
    throw std::runtime_error("Gamma must belong to [0, 1]");
  }

  const auto train_size = train.size();
  const auto whole_mini_batches_number = train_size / cfg.mini_batch_size;
  const auto remainder_mini_batch_size = train_size % cfg.mini_batch_size;

  auto [delta_weights_ema, delta_biases_ema] = CreateParameters(0);
  auto train_shuffled = std::vector(train.begin(), train.end());
  auto metric = CreateMetric(cfg);
  for (std::size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
    std::shuffle(train_shuffled.begin(), train_shuffled.end(), generator_);
    auto it = train_shuffled.begin();
    for (std::size_t i = 0; i < whole_mini_batches_number; ++i) {
      auto end = it + cfg.mini_batch_size;
      UpdateSgdNag(delta_weights_ema, delta_biases_ema, it, end,
                   cfg.mini_batch_size, cfg.learning_rate, gamma);
      it = std::move(end);
    }

    if (remainder_mini_batch_size != 0) {
      UpdateSgdNag(delta_weights_ema, delta_biases_ema, it,
                   it + remainder_mini_batch_size, remainder_mini_batch_size,
                   cfg.learning_rate, gamma);
    }
    WriteMetric(metric, epoch, train, test, cfg);
  }

  return metric;
}

Metric Perceptron::SgdAdagrad(
    const std::vector<std::shared_ptr<const IData>> &train,
    const std::vector<std::shared_ptr<const IData>> &test,
    const SgdConfiguration &cfg, const double epsilon) {
  if (epsilon <= 0) {
    throw std::runtime_error("Epsilon must be strictly greater than 0");
  }

  const auto train_size = train.size();
  const auto whole_mini_batches_number = train_size / cfg.mini_batch_size;
  const auto remainder_mini_batch_size = train_size % cfg.mini_batch_size;

  auto [weights_gradient_squares_sum, biases_gradient_squares_sum] =
      CreateParameters(epsilon);
  auto train_shuffled = std::vector(train.begin(), train.end());
  auto metric = CreateMetric(cfg);
  for (std::size_t epoch = 1; epoch < cfg.epochs; ++epoch) {
    std::shuffle(train_shuffled.begin(), train_shuffled.end(), generator_);
    auto it = train_shuffled.begin();
    for (std::size_t i = 0; i < whole_mini_batches_number; ++i) {
      auto end = it + cfg.mini_batch_size;
      UpdateSgdAdagrad(weights_gradient_squares_sum,
                       biases_gradient_squares_sum, it, end,
                       cfg.mini_batch_size, cfg.learning_rate);
      it = std::move(end);
    }

    if (remainder_mini_batch_size != 0) {
      UpdateSgdAdagrad(weights_gradient_squares_sum,
                       biases_gradient_squares_sum, it,
                       it + remainder_mini_batch_size,
                       remainder_mini_batch_size, cfg.learning_rate);
    }
    WriteMetric(metric, epoch, train, test, cfg);
  }

  return metric;
}

Metric Perceptron::SgdAdam(
    const std::vector<std::shared_ptr<const IData>> &train,
    const std::vector<std::shared_ptr<const IData>> &test,
    const SgdConfiguration &cfg, const double beta1, const double beta2,
    const double epsilon) {
  if (beta1 < 0 || beta1 > 1) {
    throw std::runtime_error("Beta1 must belong to [0, 1]");
  }
  if (beta2 < 0 || beta2 > 1) {
    throw std::runtime_error("Beta2 must belong to [0, 1]");
  }
  if (epsilon <= 0) {
    throw std::runtime_error("Epsilon must be strictly greater than 0");
  }

  const auto train_size = train.size();
  const auto whole_mini_batches_number = train_size / cfg.mini_batch_size;
  const auto remainder_mini_batch_size = train_size % cfg.mini_batch_size;

  auto [weights_gradient_ema, biases_gradient_ema] = CreateParameters(0);
  auto [weights_squared_gradient_ema, biases_squared_gradient_ema] =
      CreateParameters(epsilon);
  auto train_shuffled = std::vector(train.begin(), train.end());
  auto metric = CreateMetric(cfg);
  for (std::size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
    std::shuffle(train_shuffled.begin(), train_shuffled.end(), generator_);
    auto it = train_shuffled.begin();
    for (std::size_t i = 0; i < whole_mini_batches_number; ++i) {
      auto end = it + cfg.mini_batch_size;
      UpdateSgdAdam(weights_gradient_ema, biases_gradient_ema,
                    weights_squared_gradient_ema, biases_squared_gradient_ema,
                    it, end, cfg.mini_batch_size, epoch, cfg.learning_rate,
                    beta1, beta2);
      it = std::move(end);
    }

    if (remainder_mini_batch_size != 0) {
      UpdateSgdAdam(weights_gradient_ema, biases_gradient_ema,
                    weights_squared_gradient_ema, biases_squared_gradient_ema,
                    it, it + remainder_mini_batch_size,
                    remainder_mini_batch_size, epoch, cfg.learning_rate, beta1,
                    beta2);
    }
    WriteMetric(metric, epoch, train, test, cfg);
  }

  return metric;
}

template <typename Iter>
void Perceptron::UpdateSgd(const Iter mini_batch_begin,
                           const Iter mini_batch_end,
                           const std::size_t mini_batch_size,
                           const double learning_rate) {
  auto [weights_gradient, biases_gradient] =
      GradientWrtParameters(mini_batch_begin, mini_batch_end, mini_batch_size);

  for (std::size_t i = 0; i < connections_number_; ++i) {
    weights_[i] -= learning_rate * weights_gradient[i];
    biases_[i] -= learning_rate * biases_gradient[i];
  }
}

template <typename Iter>
void Perceptron::UpdateSgdNag(std::vector<Eigen::MatrixXd> &delta_weights_ema,
                              std::vector<Eigen::VectorXd> &delta_biases_ema,
                              const Iter mini_batch_begin,
                              const Iter mini_batch_end,
                              const std::size_t mini_batch_size,
                              const double learning_rate, const double gamma) {
  for (std::size_t i = 0; i < connections_number_; ++i) {
    weights_[i] -= gamma * delta_weights_ema[i];
    biases_[i] -= gamma * delta_biases_ema[i];
  }

  auto [weights_gradient, biases_gradient] =
      GradientWrtParameters(mini_batch_begin, mini_batch_end, mini_batch_size);

  for (std::size_t i = 0; i < connections_number_; ++i) {
    const auto saved_delta_weights_ema = gamma * delta_weights_ema[i];
    delta_weights_ema[i] =
        saved_delta_weights_ema + learning_rate * weights_gradient[i];
    weights_[i] += saved_delta_weights_ema - delta_weights_ema[i];

    const auto saved_delta_biases_ema = gamma * delta_biases_ema[i];
    delta_biases_ema[i] =
        saved_delta_biases_ema + learning_rate * biases_gradient[i];
    biases_[i] += saved_delta_biases_ema - delta_biases_ema[i];
  }
}

template <typename Iter>
void Perceptron::UpdateSgdAdagrad(
    std::vector<Eigen::MatrixXd> &weights_gradient_squares_sum,
    std::vector<Eigen::VectorXd> &biases_gradient_squares_sum,
    const Iter mini_batch_begin, const Iter mini_batch_end,
    const std::size_t mini_batch_size, const double learning_rate) {
  auto [weights_gradient, biases_gradient] =
      GradientWrtParameters(mini_batch_begin, mini_batch_end, mini_batch_size);

  for (std::size_t i = 0; i < connections_number_; ++i) {
    weights_gradient_squares_sum[i] +=
        weights_gradient[i].array().pow(2).matrix();
    weights_[i] -= learning_rate / weights_gradient_squares_sum[i].lpNorm<2>() *
                   weights_gradient[i];

    biases_gradient_squares_sum[i] +=
        biases_gradient[i].array().pow(2).matrix();
    biases_[i] -= learning_rate / biases_gradient_squares_sum[i].lpNorm<2>() *
                  biases_gradient[i];
  }
}

template <typename Iter>
void Perceptron::UpdateSgdAdam(
    std::vector<Eigen::MatrixXd> &weights_gradient_ema,
    std::vector<Eigen::VectorXd> &biases_gradient_ema,
    std::vector<Eigen::MatrixXd> &weights_squared_gradient_ema,
    std::vector<Eigen::VectorXd> &biases_squared_gradient_ema,
    const Iter mini_batch_begin, const Iter mini_batch_end,
    const std::size_t mini_batch_size, const std::size_t epoch,
    const double learning_rate, const double beta1, const double beta2) {
  auto [weights_gradient, biases_gradient] =
      GradientWrtParameters(mini_batch_begin, mini_batch_end, mini_batch_size);

  for (std::size_t i = 0; i < connections_number_; ++i) {
    // Надеюсь, комплиятор догадается вынести available expressions - лично мне
    // лень.
    weights_gradient_ema[i] =
        beta1 * weights_gradient_ema[i] + (1 - beta1) * weights_gradient[i];
    biases_gradient_ema[i] =
        beta1 * biases_gradient_ema[i] + (1 - beta1) * biases_gradient[i];

    const auto adjusted_weights_gradient_ema =
        weights_gradient_ema[i] / (1 - std::pow(beta1, epoch));
    const auto adjusted_biases_gradient_ema =
        biases_gradient_ema[i] / (1 - std::pow(beta1, epoch));

    weights_squared_gradient_ema[i] =
        beta2 * weights_squared_gradient_ema[i] +
        (1 - beta2) * weights_gradient[i].array().pow(2).matrix();
    biases_squared_gradient_ema[i] =
        beta2 * biases_squared_gradient_ema[i] +
        (1 - beta2) * biases_gradient[i].array().pow(2).matrix();

    const auto adjusted_weights_squared_gradient_ema =
        weights_squared_gradient_ema[i] / (1 - std::pow(beta2, epoch));
    const auto adjusted_biases_squared_gradient_ema =
        biases_squared_gradient_ema[i] / (1 - std::pow(beta2, epoch));

    weights_[i] -= learning_rate /
                   adjusted_weights_squared_gradient_ema.lpNorm<2>() *
                   adjusted_weights_gradient_ema;
    biases_[i] -= learning_rate /
                  adjusted_biases_squared_gradient_ema.lpNorm<2>() *
                  adjusted_biases_gradient_ema;
  }
}

Perceptron::Parameters Perceptron::CreateParameters(
    const double initial_value) const {
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

template <typename Iter>
Perceptron::Parameters Perceptron::GradientWrtParameters(
    const Iter mini_batch_begin, const Iter mini_batch_end,
    const std::size_t mini_batch_size) const {
  auto [weights_gradient, biases_gradient] = CreateParameters(0);

  for (auto it = mini_batch_begin; it != mini_batch_end; ++it) {
    const auto &data = **it;
    const auto [weights_gradient_contribution, biases_gradient_contribution] =
        Backpropagation(data.GetX(), data.GetY());
    for (std::size_t i = 0; i < connections_number_; ++i) {
      weights_gradient[i] += weights_gradient_contribution[i];
      biases_gradient[i] += biases_gradient_contribution[i];
    }
  }

  const auto factor = 1.0 / mini_batch_size;
  for (std::size_t i = 0; i < connections_number_; ++i) {
    weights_gradient[i] *= factor;
    biases_gradient[i] *= factor;
  }

  return {weights_gradient, biases_gradient};
}

Perceptron::Parameters Perceptron::Backpropagation(
    const Eigen::VectorXd &x, const Eigen::VectorXd &y) const {
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
Perceptron::FeedforwardDetailed(const Eigen::VectorXd &x) const {
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

Metric Perceptron::CreateMetric(const SgdConfiguration &cfg) const {
  auto metric = Metric{};
  if (cfg.monitor_train_cost) {
    metric.train_cost.reserve(cfg.epochs);
  }
  if (cfg.monitor_train_accuracy) {
    metric.train_accuracy.reserve(cfg.epochs);
  }
  if (cfg.monitor_test_cost) {
    metric.test_cost.reserve(cfg.epochs);
  }
  if (cfg.monitor_test_accuracy) {
    metric.test_accuracy.reserve(cfg.epochs);
  }
  return metric;
}

void Perceptron::WriteMetric(
    Metric &metric, const std::size_t epoch,
    const std::vector<std::shared_ptr<const IData>> &train,
    const std::vector<std::shared_ptr<const IData>> &test,
    const SgdConfiguration &cfg) const {
  std::stringstream oss;
  oss << "Epoch " << epoch << ";";
  if (cfg.monitor_train_cost) {
    const auto train_cost = CalculateCost(train.begin(), train.end());
    metric.train_cost.push_back(train_cost);
    oss << " train cost: " << train_cost << ";";
  }
  if (cfg.monitor_train_accuracy) {
    const auto train_accuracy = CalculateAccuracy(train.begin(), train.end());
    metric.train_accuracy.push_back(train_accuracy);
    oss << " train accuracy: " << train_accuracy << "/" << train.size() << ";";
  }
  if (cfg.monitor_test_cost) {
    const auto test_cost = CalculateCost(test.begin(), test.end());
    metric.test_cost.push_back(CalculateCost(test.begin(), test.end()));
    oss << " test cost: " << test_cost << ";";
  }
  if (cfg.monitor_test_accuracy) {
    const auto test_accuracy = CalculateAccuracy(test.begin(), test.end());
    metric.test_accuracy.push_back(test_accuracy);
    oss << " test accuracy: " << test_accuracy << "/" << test.size() << ";";
  }
  spdlog::info(oss.str());
}

template <typename Iter>
std::size_t Perceptron::CalculateAccuracy(const Iter begin,
                                          const Iter end) const {
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
double Perceptron::CalculateCost(const Iter begin, const Iter end) const {
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
