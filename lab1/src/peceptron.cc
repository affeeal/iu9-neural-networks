#include <iterator>

#include "perceptron.h"

namespace lab1 {

Perceptron::Perceptron(
    std::unique_ptr<ICostFunction>&& cost_function,
    std::vector<std::unique_ptr<IActivationFunction>>&& activation_functions,
    const std::vector<std::size_t>& layers_sizes)
    : cost_function_(std::move(cost_function)),
      layers_number_(layers_sizes.size()),
      connections_number_(layers_number_ - 1),
      activation_functions_(std::move(activation_functions)) {
  if (layers_number_ < 2) {
    throw std::runtime_error("Perceptron must have at least two layers");
  }

  if (activation_functions_.size() != connections_number_) {
    throw std::runtime_error(
        "Activation functions number must be equal to layers number minus "
        "one");
  }

  weights_.reserve(connections_number_);
  biases_.reserve(connections_number_);
  for (std::size_t i = 0; i < connections_number_; ++i) {
    weights_.push_back(
        Eigen::MatrixXd::Random(layers_sizes[i + 1], layers_sizes[i]));
    biases_.push_back(Eigen::VectorXd::Random(layers_sizes[i + 1]));
  }
}

Eigen::VectorXd Perceptron::Feedforward(const Eigen::VectorXd& x) const {
  auto activation = x;
  for (std::size_t i = 0; i < connections_number_; ++i) {
    activation =
        activation_functions_[i]->Apply(weights_[i] * activation + biases_[i]);
  }
  return activation;
}

void Perceptron::StochasticGradientSearch(
    const std::vector<std::shared_ptr<const IData>>& training,
    const std::size_t epochs, const std::size_t mini_batch_size,
    const double eta,
    const std::vector<std::shared_ptr<const IData>>& testing) {
  const std::size_t training_size = training.size();
  const std::size_t whole_mini_batches = training_size / mini_batch_size;
  const std::size_t remainder_mini_batch_size = training_size % mini_batch_size;
  const std::size_t testing_size = testing.size();

  auto training_shuffled = std::vector(training.begin(), training.end());

  for (std::size_t i = 0; i < epochs; ++i) {
    std::shuffle(training_shuffled.begin(), training_shuffled.end(),
                 generator_);
    auto it = training_shuffled.begin();
    for (std::size_t i = 0; i < whole_mini_batches; ++i) {
      auto end = it + mini_batch_size;
      UpdateMiniBatch(it, end, mini_batch_size, eta);
      it = std::move(end);
    }
    UpdateMiniBatch(it, it + remainder_mini_batch_size, mini_batch_size, eta);
    std::cout << "Epoch " << i << ": " << Evaluate(testing) << "/"
              << testing_size << '\n';
  }
}

template <typename Iter>
void Perceptron::UpdateMiniBatch(const Iter mini_batch_begin,
                                 const Iter mini_batch_end,
                                 const std::size_t mini_batch_size,
                                 const double eta) {
  auto nabla_weights = std::vector<Eigen::MatrixXd>{};
  nabla_weights.reserve(connections_number_);
  for (auto&& w : weights_) {
    nabla_weights.push_back(Eigen::MatrixXd::Zero(w.rows(), w.cols()));
  }

  auto nabla_biases = std::vector<Eigen::VectorXd>{};
  nabla_biases.reserve(connections_number_);
  for (auto&& b : biases_) {
    nabla_biases.push_back(Eigen::VectorXd::Zero(b.size()));
  }

  for (auto it = mini_batch_begin; it != mini_batch_end; ++it) {
    const auto& data = **it;
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
Perceptron::Backpropagation(const Eigen::VectorXd& x,
                            const Eigen::VectorXd& y) {
  const auto [zs, activations] = FeedforwardPass(x);
  assert(zs.size() == connections_number_);
  assert(activations.size() == layers_number_);

  auto delta = static_cast<Eigen::VectorXd>(
      cost_function_->ActivationsPrime(y, activations.back()).array() *
      activation_functions_.back()->Prime(zs.back()).array());

  auto nabla_weights_reversed = std::vector<Eigen::MatrixXd>{};
  nabla_weights_reversed.reserve(connections_number_);
  nabla_weights_reversed.push_back(
      delta * std::prev(activations.cend(), 2)->transpose());

  auto nabla_biases_reversed = std::vector<Eigen::VectorXd>{};
  nabla_biases_reversed.reserve(connections_number_);
  nabla_biases_reversed.push_back(delta);

  for (int i = connections_number_ - 2; i >= 0; --i) {
    delta = (weights_[i + 1].transpose() * delta).array() *
            activation_functions_[i]->Prime(zs[i]).array();
    nabla_weights_reversed.push_back(delta * activations[i].transpose());
    nabla_biases_reversed.push_back(delta);
  }

  return {{std::make_move_iterator(nabla_weights_reversed.rbegin()),
           std::make_move_iterator(nabla_weights_reversed.rend())},
          {std::make_move_iterator(nabla_biases_reversed.rbegin()),
           std::make_move_iterator(nabla_biases_reversed.rend())}};
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
Perceptron::FeedforwardPass(const Eigen::VectorXd& x) {
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

std::size_t Perceptron::Evaluate(
    const std::vector<std::shared_ptr<const IData>>& data) {
  std::size_t right_predictions = 0;
  for (auto&& instance : data) {
    Eigen::Index max_expected, max_actual;
    instance->GetY().maxCoeff(&max_expected);
    Feedforward(instance->GetX()).maxCoeff(&max_actual);
    if (max_expected == max_actual) {
      ++right_predictions;
    }
  }
  return right_predictions;
}

}  // namespace lab1
