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

const std::string kDefaultTestPath = "../../datasets/MNIST_CSV/test.csv";
const std::string kDefaultTrainPath = "../../datasets/MNIST_CSV/train.csv";

void RunLeakyReluSoftmaxCrossEntropy() {
  constexpr std::size_t kHiddenLayerSize = 40;
  constexpr static auto kCfg = nn::SgdConfiguration{
      .epochs = 20,
      .mini_batch_size = 10,
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
  const auto metrics = perceptron.Sgd(train, test, kCfg);

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

void RunGeneticAlgorithmSgd() {
  /*
   * Train cost: 0.0964943;
   * Train accuracy: 48532/50000;
   * Test cost: 0.14492;
   * Test accuracy: 9580/10000;
   *
   * Learning rate: 0.0959074;
   * Epochs: 100;
   * Mini-batch size: 100;
   * Hidden layers: 1;
   * Neurons per hidden layer: 28
   */

  auto data_supplier = std::make_unique<nn::DataSupplier>(
      kDefaultTrainPath, kDefaultTestPath, 0.0, 1.0);
  auto fitness_function =
      std::make_unique<nn::SgdFitness>(std::move(data_supplier));
  const auto segments = std::vector<nn::Segment>{
      {0.001, 1},  // kLearningRate
      {100, 100},  // kEpochs
      {100, 100},  // kMiniBatchSize
      {0, 4},      // kHiddenLayer
      {10, 40},    // kNeuronsPerHiddenLayer
  };
  const auto cfg = nn::GeneticAlgorithm::Configuration{
      .populations_number = 10,
      .population_size = 60,
      .crossover_proportion = 0.4,
      .mutation_proportion = 0.15,
  };
  auto genetic_algorithm = nn::GeneticAlgorithm(
      std::move(fitness_function),
      nn::ChromosomeSubclass::kSgdHyperparametersKit, segments, cfg);
  genetic_algorithm.Run();
}

void RunGeneticAlgorithmSgdNag() {
  /*
   * Train cost: 0.0957484;
   * Train accuracy: 48562/50000;
   * Test cost: 0.146654;
   * Test accuracy: 9565/10000;
   *
   * Learning rate: 0.0100863;
   * Epochs: 100;
   * Mini-batch size: 100;
   * Hidden layers: 1;
   * Neurons per hidden layer: 28
   */

  auto data_supplier = std::make_unique<nn::DataSupplier>(
      kDefaultTrainPath, kDefaultTestPath, 0.0, 1.0);
  auto fitness_function =
      std::make_unique<nn::SgdNagFitness>(std::move(data_supplier));
  const auto segments = std::vector<nn::Segment>{
      {0.001, 1},  // kLearningRate
      {100, 100},  // kEpochs
      {100, 100},  // kMiniBatchSize
      {0, 4},      // kHiddenLayer
      {10, 40},    // kNeuronsPerHiddenLayer
  };
  const auto cfg = nn::GeneticAlgorithm::Configuration{
      .populations_number = 10,
      .population_size = 60,
      .crossover_proportion = 0.4,
      .mutation_proportion = 0.15,
  };
  auto genetic_algorithm = nn::GeneticAlgorithm(
      std::move(fitness_function),
      nn::ChromosomeSubclass::kSgdHyperparametersKit, segments, cfg);
  genetic_algorithm.Run();
}

void RunGeneticAlgorithmSgdAdagrad() {
  /*
   * Train cost: 0.181089;
   * Train accuracy: 47361/50000;
   * Test cost: 0.214604;
   * Test accuracy: 9395/10000;
   *
   * Learning rate: 0.936544;
   * Epochs: 100;
   * Mini-batch size: 100;
   * Hidden layers: 1;
   * Neurons per hidden layer: 36
   */

  auto data_supplier = std::make_unique<nn::DataSupplier>(
      kDefaultTrainPath, kDefaultTestPath, 0.0, 1.0);
  auto fitness_function =
      std::make_unique<nn::SgdAdagradFitness>(std::move(data_supplier));
  const auto segments = std::vector<nn::Segment>{
      {0.001, 1},  // kLearningRate
      {100, 100},  // kEpochs
      {100, 100},  // kMiniBatchSize
      {0, 4},      // kHiddenLayer
      {10, 40},    // kNeuronsPerHiddenLayer
  };
  const auto cfg = nn::GeneticAlgorithm::Configuration{
      .populations_number = 10,
      .population_size = 60,
      .crossover_proportion = 0.4,
      .mutation_proportion = 0.15,
  };
  auto genetic_algorithm = nn::GeneticAlgorithm(
      std::move(fitness_function),
      nn::ChromosomeSubclass::kSgdHyperparametersKit, segments, cfg);
  genetic_algorithm.Run();
}

void RunGeneticAlgorithmSgdAdam() {
  auto data_supplier = std::make_unique<nn::DataSupplier>(
      kDefaultTrainPath, kDefaultTestPath, 0.0, 1.0);
  auto fitness_function =
      std::make_unique<nn::SgdAdamFitness>(std::move(data_supplier));
  const auto segments = std::vector<nn::Segment>{
      {0.001, 1},  // kLearningRate
      {100, 100},  // kEpochs
      {100, 100},  // kMiniBatchSize
      {0, 4},      // kHiddenLayer
      {10, 40},    // kNeuronsPerHiddenLayer
  };
  const auto cfg = nn::GeneticAlgorithm::Configuration{
      .populations_number = 10,
      .population_size = 60,
      .crossover_proportion = 0.4,
      .mutation_proportion = 0.15,
  };
  auto genetic_algorithm = nn::GeneticAlgorithm(
      std::move(fitness_function),
      nn::ChromosomeSubclass::kSgdHyperparametersKit, segments, cfg);
  genetic_algorithm.Run();
}

}  // namespace

int main() {
  RunGeneticAlgorithmSgd();
  RunGeneticAlgorithmSgdNag();
  RunGeneticAlgorithmSgdAdagrad();
  RunGeneticAlgorithmSgdAdam();
}
