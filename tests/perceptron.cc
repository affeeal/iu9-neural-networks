#include "perceptron.h"
#if HOMEWORK == 1
#include "util.h"
#endif

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <set>

#include "check.h"
#include "data_supplier.h"
#include "output_layer.h"

namespace {

struct Sample : nn::IData {
  Eigen::VectorXd x = Eigen::Vector2d(0.25, -0.75);
  Eigen::VectorXd y = Eigen::Vector2d(0.0, 1.0);
  const Eigen::VectorXd& GetX() const override { return x; }
  const Eigen::VectorXd& GetY() const override { return y; }
  std::string_view ToString() const override { return "sample"; }
};
using Dataset = std::vector<std::shared_ptr<const nn::IData>>;
#if HOMEWORK == 4
using Config = nn::SgdConfiguration;
#else
using Config = nn::Config;
#endif

nn::Metric Train(nn::Perceptron& p, const Dataset& data, const Config& cfg,
                 int method = 0) {
#if HOMEWORK == 4
  if (method == 1) return p.SgdNag(data, data, cfg, 0.9);
  if (method == 2) return p.SgdAdagrad(data, data, cfg, 1e-8);
  if (method == 3) return p.SgdAdam(data, data, cfg, 0.9, 0.999, 1e-8);
  return p.Sgd(data, data, cfg);
#else
  (void)method;
  return p.StochasticGradientSearch(data, data, cfg);
#endif
}

nn::Perceptron Linear() {
  std::vector<std::unique_ptr<nn::IActivationFunction>> activations;
  activations.push_back(std::make_unique<nn::Linear>());
  return nn::Perceptron(std::make_unique<nn::MSE>(), std::move(activations),
                        {2, 2}, 42);
}

void CheckOptimizers() {
  auto sample = std::make_shared<Sample>();
  // Three identical examples remove shuffle dependence, including a partial
  // batch.
  const Dataset data(3, sample);
  constexpr double lr = 0.03, epsilon = 1e-8, beta1 = 0.9, beta2 = 0.999;
  const Config cfg{2, 2, lr, true, true, true, true};
  const int methods = HOMEWORK == 4 ? 4 : 1;
  for (int method = 0; method < methods; ++method) {
    auto network = Linear();
    Eigen::Matrix<double, 2, 3> parameters;
    parameters.leftCols<2>() = network.get_weights()[0];
    parameters.col(2) = network.get_biases()[0];
    Eigen::Matrix<double, 2, 3> first = Eigen::Matrix<double, 2, 3>::Zero();
    auto second = first;
    Eigen::Vector3d input(sample->x[0], sample->x[1], 1);
    for (int step = 1; step <= 4; ++step) {
      const Eigen::Matrix<double, 2, 3> lookahead =
          method == 1 ? (parameters - beta1 * first).eval() : parameters;
      const Eigen::Matrix<double, 2, 3> gradient =
          (lookahead * input - sample->y) * input.transpose();
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
          const double g = gradient(i, j);
          double update = lr * g;
          if (method == 1) {
            first(i, j) = beta1 * first(i, j) + lr * g;
            update = first(i, j);
          } else if (method == 2) {
            second(i, j) += g * g;
            update = lr * g / (std::sqrt(second(i, j)) + epsilon);
          } else if (method == 3) {
            first(i, j) = beta1 * first(i, j) + (1 - beta1) * g;
            second(i, j) = beta2 * second(i, j) + (1 - beta2) * g * g;
            update = lr * (first(i, j) / (1 - std::pow(beta1, step))) /
                     (std::sqrt(second(i, j) / (1 - std::pow(beta2, step))) +
                      epsilon);
          }
          parameters(i, j) -= update;
        }
      }
    }
    Train(network, data, cfg, method);
    Near((network.get_weights()[0] - parameters.leftCols<2>()).norm(), 0,
         1e-12);
    Near((network.get_biases()[0] - parameters.col(2)).norm(), 0, 1e-12);
  }
}

void CheckBackpropagation() {
  std::vector<std::unique_ptr<nn::IActivationFunction>> activations;
  activations.push_back(std::make_unique<nn::Tanh>());
  activations.push_back(std::make_unique<nn::Softmax>());
  nn::Perceptron network(std::make_unique<nn::CrossEntropy>(),
                         std::move(activations), {2, 3, 2}, 42);
  auto weights = network.get_weights();
  auto biases = network.get_biases();
  const auto before_w = weights;
  const auto before_b = biases;
  const auto sample = std::make_shared<Sample>();
  nn::Tanh tanh;
  nn::Softmax softmax;
  nn::CrossEntropy ce;
  const auto loss = [&] {
    const Eigen::VectorXd hidden =
        tanh.Apply(weights[0] * sample->x + biases[0]);
    return nn::OutputCost(softmax, ce, sample->y,
                          weights[1] * hidden + biases[1]);
  };
  constexpr double h = 1e-5, lr = 0.01;
  auto dw = weights;
  auto db = biases;
  const auto derivative = [&](double& parameter) {
    const double saved = parameter;
    parameter = saved + h;
    const double plus = loss();
    parameter = saved - h;
    const double minus = loss();
    parameter = saved;
    return (plus - minus) / (2 * h);
  };
  for (std::size_t layer = 0; layer < weights.size(); ++layer) {
    for (Eigen::Index i = 0; i < weights[layer].size(); ++i)
      dw[layer].data()[i] = derivative(weights[layer].data()[i]);
    for (Eigen::Index i = 0; i < biases[layer].size(); ++i)
      db[layer][i] = derivative(biases[layer][i]);
  }
  Train(network, {sample}, Config{1, 1, lr, true, true, true, true});
  for (std::size_t layer = 0; layer < weights.size(); ++layer) {
    Near((network.get_weights()[layer] - before_w[layer] + lr * dw[layer])
             .norm(),
         0, 1e-9);
    Near(
        (network.get_biases()[layer] - before_b[layer] + lr * db[layer]).norm(),
        0, 1e-9);
  }
  auto linear = Linear();
  Throws([&] {
    Train(linear, {sample}, Config{1, 0, lr, false, false, false, false});
  });
  Throws(
      [&] { Train(linear, {}, Config{1, 1, lr, false, false, false, false}); });
  Throws([&] { linear.Feedforward(Eigen::VectorXd::Zero(3)); });
  const auto bad = std::make_shared<Sample>();
  bad->y.resize(3);
  Throws([&] {
    Train(linear, {bad}, Config{1, 1, lr, false, false, false, false});
  });
}

#if HOMEWORK != 1
#if HOMEWORK == 2
using Supplier = hw2::DataSupplier;
#else
using Supplier = nn::DataSupplier;
#endif
struct Csv {
  std::filesystem::path path =
      "mnist-test-hw" + std::to_string(HOMEWORK) + ".csv";
  ~Csv() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
  void Write(const std::string& text) const { std::ofstream(path) << text; }
};
void CheckData() {
  Csv csv;
  std::string row = "3";
  for (int i = 0; i < 784; ++i) row += ",255";
  csv.Write(row + "\r\n" + row + "\n" + row + "\n");
  const Supplier supplier(csv.path, csv.path, 0, 1, 1);
#if HOMEWORK == 2
  const auto train = supplier.GetTrainingData();
#else
  const auto train = supplier.GetTrainData();
#endif
  CHECK(train.size() == 2 && supplier.GetValidationData().size() == 1);
  Near(train[0]->GetX().sum(), 784);
  Near(train[0]->GetY()[3], 1);
  Throws([&] { Supplier invalid(csv.path, csv.path, 0, 1, 3); });
  for (const auto& text : std::vector<std::string>{
           "", "label,pixel\n", "0,1\n", row + ",0\n", "10" + row.substr(1),
           row.substr(0, row.rfind(',')) + ",256",
           row.substr(0, row.rfind(',')) + ",nan",
           row.substr(0, row.rfind(',')) + ",-1"}) {
    csv.Write(text);
    Throws([&] { Supplier invalid(csv.path, csv.path, 0, 1, 1); });
  }
}
#else
void CheckData() {
  const hw1::DataSupplier supplier(0, 1);
  CHECK(hw1::GeneratePowerset(std::vector<int>{}).size() == 1);
  CHECK(hw1::GeneratePowerset(std::vector<int>{1, 2}).size() == 4);
  Throws([] {
    hw1::GeneratePowerset(
        std::vector<int>(std::numeric_limits<std::size_t>::digits));
  });
  const auto train = supplier.GetTrainingData();
  const auto validation = supplier.GetValidationData();
  const auto test = supplier.GetTestingData();
  CHECK(train.size() == 160 && validation.size() == 60 && test.size() == 100);
  std::set<std::string> seen;
  for (const auto* data : {&train, &validation, &test}) {
    for (const auto& sample : *data) {
      CHECK(sample->GetX().size() == 20 && sample->GetY().size() == 20);
      Near(sample->GetY().sum(), 1);
      std::string key(sample->ToString());
      for (double value : sample->GetX()) key += value == 0 ? '0' : '1';
      CHECK(seen.insert(key).second);
    }
  }
}
#endif
}  // namespace

int main() {
  spdlog::set_level(spdlog::level::off);
  CheckOptimizers();
  CheckBackpropagation();
  CheckData();
}
