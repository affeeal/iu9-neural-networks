#include <matplot/matplot.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <memory>

#include "activation_function.h"
#include "chromosome.h"
#include "cost_function.h"
#include "data_supplier.h"
#include "fitness_function.h"
#include "genetic_algorithm.h"
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
      .monitor_train_cost = true,
      .monitor_train_accuracy = true,
      .monitor_test_cost = true,
      .monitor_test_accuracy = true,
  };

  const auto data_supplier =
      nn::DataSupplier(kDefaultTrainPath, kDefaultTestPath, 0.0, 1.0);
  const auto train = data_supplier.GetTrainData();
  const auto test = data_supplier.GetTestData();

  auto cost_function = std::make_unique<nn::CrossEntropy>();
  auto activation_functions =
      std::vector<std::unique_ptr<nn::IActivationFunction>>{};
  activation_functions.push_back(std::make_unique<nn::LeakyReLU>(0.01));
  activation_functions.push_back(std::make_unique<nn::Softmax>());
  const auto layers_sizes = std::vector<std::size_t>{
      data_supplier.GetInputLayerSize(), kHiddenLayerSize,
      data_supplier.GetOutputLayerSize()};

  auto perceptron = nn::Perceptron(
      std::move(cost_function), std::move(activation_functions), layers_sizes);
  const auto metrics = perceptron.SgdAdam(train, test, kCfg, 0.9, 0.999, 1e-8);

  matplot::title("Leaky ReLU, Softmax + Cross-entropy train, test cost");
  matplot::plot(metrics.train_cost)->display_name("Train data");
  matplot::hold(matplot::on);
  matplot::plot(metrics.test_cost)->display_name("Test data");
  matplot::hold(matplot::off);
  matplot::legend({});
  matplot::xlabel("Epochs");
  matplot::ylabel("Cost");
  matplot::show();

  matplot::title("Leaky ReLU, Softmax + Cross-entropy train, test accuracy");
  matplot::plot(metrics.train_accuracy)->display_name("Train data");
  matplot::hold(matplot::on);
  matplot::plot(metrics.test_accuracy)->display_name("Test data");
  matplot::hold(matplot::off);
  matplot::legend({});
  matplot::xlabel("Epochs");
  matplot::ylabel("Hit");
  matplot::show();
}

void RunGeneticAlgorithm() {
  auto data_supplier = std::make_unique<nn::DataSupplier>(
      kDefaultTrainPath, kDefaultTestPath, 0.0, 1.0);
  auto fitness_function =
      std::make_unique<nn::AccuracyOnTestData>(std::move(data_supplier));
  const auto segments = std::vector<nn::Segment>{
      {0.01, 0.03},  // kLearningRate
      {5, 50},       // kEpochs
      {1, 100},      // kMiniBatchSize
      {0, 3},        // kHiddenLayer
      {10, 50},      // kNeuronsPerHiddenLayer
  };
  const auto cfg = nn::GeneticAlgorithm::Configuration{
      .populations_number = 5,
      .population_size = 45,
      .crossover_proportion = 0.4,
      .mutation_proportion = 0.15,
  };
  auto genetic_algorithm = nn::GeneticAlgorithm(
      std::move(fitness_function),
      nn::ChromosomeSubclass::kSgdHyperparametersKit, segments, cfg);
  genetic_algorithm.Run();
}

}  // namespace

int main() { RunGeneticAlgorithm(); }
