#ifdef NN_ENABLE_PLOTS
#include <matplot/matplot.h>
#endif
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <memory>
#include <string_view>

#include "activation_function.h"
#include "chromosome.h"
#include "cost_function.h"
#include "data_supplier.h"
#include "fitness_function.h"
#include "genetic_algorithm.h"
#include "perceptron.h"

namespace {

std::string test_path;
std::string train_path;

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

  const auto data_supplier = nn::DataSupplier(train_path, test_path, 0.0, 1.0);
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

#ifdef NN_ENABLE_PLOTS
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
#endif
}

void RunGeneticAlgorithmSgd() {
  auto data_supplier =
      std::make_unique<nn::DataSupplier>(train_path, test_path, 0.0, 1.0);
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
  auto data_supplier =
      std::make_unique<nn::DataSupplier>(train_path, test_path, 0.0, 1.0);
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
  auto data_supplier =
      std::make_unique<nn::DataSupplier>(train_path, test_path, 0.0, 1.0);
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
  auto data_supplier =
      std::make_unique<nn::DataSupplier>(train_path, test_path, 0.0, 1.0);
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

int main(int argc, char* argv[]) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout << "Usage: hw4 TRAIN.csv TEST.csv {train|sgd|nag|adagrad|adam}\n";
    return 0;
  }
  if (argc != 4) {
    std::cerr << "Usage: hw4 TRAIN.csv TEST.csv {train|sgd|nag|adagrad|adam}\n";
    return 1;
  }
  train_path = argv[1];
  test_path = argv[2];
  try {
    const std::string_view optimizer = argv[3];
    if (optimizer == "train")
      RunLeakyReluSoftmaxCrossEntropy();
    else if (optimizer == "sgd")
      RunGeneticAlgorithmSgd();
    else if (optimizer == "nag")
      RunGeneticAlgorithmSgdNag();
    else if (optimizer == "adagrad")
      RunGeneticAlgorithmSgdAdagrad();
    else if (optimizer == "adam")
      RunGeneticAlgorithmSgdAdam();
    else
      throw std::invalid_argument("Unknown optimizer");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
