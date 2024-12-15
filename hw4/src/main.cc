#include <matplot/matplot.h>

#include <memory>

#include "activation_function.h"
#include "cost_function.h"
#include "data_supplier.h"
#include "perceptron.h"

namespace {

// Sgd: lr=0.03, 7472-8832
// SgdNag: lr=0.03, gamma=0.9, 8701-9353
// SgdAdagrad: lr=5, epsilon=1e-8, 8483-9274
// SgdAdam: lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, 8110-9458

const std::string kDefaultTestPath = "../../mnist/mnist_test.csv";
const std::string kDefaultTrainPath = "../../mnist/mnist_train.csv";

void RunLeakyReluSoftmaxCrossEntropy() {
  constexpr std::size_t kHiddenLayerSize = 40;
  constexpr static auto kCfg = nn::SgdConfiguration{
      .epochs = 20,
      .mini_batch_size = 100,
      .learning_rate = 0.1,
      .monitor_training_cost = true,
      .monitor_training_accuracy = true,
      .monitor_testing_cost = true,
      .monitor_testing_accuracy = true,
  };

  const auto data_supplier =
      hw4::DataSupplier(kDefaultTrainPath, kDefaultTestPath, 0.0, 1.0);
  const auto training = data_supplier.GetTrainingData();
  const auto testing = data_supplier.GetTestingData();

  auto cost_function = std::make_unique<nn::CrossEntropy>();
  auto activation_functions =
      std::vector<std::unique_ptr<nn::IActivationFunction>>{};
  activation_functions.push_back(std::make_unique<nn::LeakyReLU>(0.01));
  activation_functions.push_back(std::make_unique<nn::Softmax>());
  const auto layers_sizes = std::vector<std::size_t>{
      hw4::kScanSize, kHiddenLayerSize, hw4::kDigitsNumber};

  auto perceptron = nn::Perceptron(
      std::move(cost_function), std::move(activation_functions), layers_sizes);
  const auto metrics =
      perceptron.SgdAdam(training, testing, kCfg, 0.9, 0.999, 1e-8);

  matplot::title("Leaky ReLU, Softmax + Cross-entropy training, testing cost");
  matplot::plot(metrics.training_cost)->display_name("Training data");
  matplot::hold(matplot::on);
  matplot::plot(metrics.testing_cost)->display_name("Testing data");
  matplot::hold(matplot::off);
  matplot::legend({});
  matplot::xlabel("Epochs");
  matplot::ylabel("Cost");
  matplot::show();

  matplot::title(
      "Leaky ReLU, Softmax + Cross-entropy training, testing accuracy");
  matplot::plot(metrics.training_accuracy)->display_name("Training data");
  matplot::hold(matplot::on);
  matplot::plot(metrics.testing_accuracy)->display_name("Testing data");
  matplot::hold(matplot::off);
  matplot::legend({});
  matplot::xlabel("Epochs");
  matplot::ylabel("Hit");
  matplot::show();
}

}  // namespace

int main(int argc, char *argv[]) { RunLeakyReluSoftmaxCrossEntropy(); }
