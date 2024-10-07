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

  // std::cout << connections_number_ << '\n';
  weights_.reserve(connections_number_);
  biases_.reserve(connections_number_);
  for (std::size_t i = 0; i < connections_number_; ++i) {
    // std::cout << layers_sizes[i + 1] << ", " << layers_sizes[i] << std::endl;
    weights_.push_back(
        Eigen::MatrixXd::Random(layers_sizes[i + 1], layers_sizes[i]));
    // std::cout << weights_.back().size() << std::endl;
    //  std::cout << weights_.back() << std::endl;
    //  std::cout << weights_.back().rows() << '\n';
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
  // std::cout << training_size << '\n';
  const std::size_t whole_mini_batches = training_size / mini_batch_size;
  // std::cout << whole_mini_batches << '\n';
  const std::size_t remainder_mini_batch_size = training_size % mini_batch_size;
  // std::cout << remainder_mini_batch_size << '\n';
  const std::size_t testing_size = testing.size();

  auto training_shuffled = std::vector(training.begin(), training.end());
  for (std::size_t i = 0; i < epochs; ++i) {
    // std::cout << i << '\n';
    std::shuffle(training_shuffled.begin(), training_shuffled.end(),
                 generator_);
    auto it = training_shuffled.begin();
    for (std::size_t i = 0; i < whole_mini_batches; ++i) {
      // std::cout << i << '\n';
      auto end = it + mini_batch_size;
      // std::cout << end - it << '\n';
      UpdateMiniBatch(it, end, mini_batch_size, eta);
      // std::cout << '?' << '\n';
      it = std::move(end);
    }
    UpdateMiniBatch(it, it + remainder_mini_batch_size, mini_batch_size, eta);

    std::cout << "Epoch " << i << ": " << Evaluate(testing) << "/"
              << testing_size << '\n';
  }
}

std::size_t Perceptron::Evaluate(
    const std::vector<std::shared_ptr<const IData>>& testing) {
  std::size_t right_predictions = 0;
  for (auto&& data : testing) {
    const auto y = Feedforward(data->GetX());
    const auto expected_max = (y.array() * data->GetY().array()).maxCoeff();
    const auto max = data->GetY().maxCoeff();
    if (max == expected_max) {
      right_predictions++;
    }
  }
  return right_predictions;
}

}  // namespace lab1
