#include "data_supplier.h"

#include <spdlog/spdlog.h>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <cassert>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "perceptron.h"

namespace nn {

namespace {

constexpr std::size_t kDigitsNumber = 10;
constexpr std::size_t kScanSize = 784;
constexpr std::size_t kColumnsCount = kScanSize + 1;

std::vector<std::shared_ptr<const nn::IData>> ReadMnistCsv(
    const std::string &filename, const double false_score,
    const double true_score) {
  static constexpr std::size_t kShadesCount = 255;

  auto file = std::ifstream(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open MNIST CSV file " + filename);
  }

  auto instances = std::vector<std::shared_ptr<const nn::IData>>{};
  auto line = std::string{};
  while (std::getline(file, line)) {
    auto result = std::vector<std::string>{};
    result.reserve(kColumnsCount);
    boost::split(result, line, boost::is_any_of(","));

    assert(result[0].size() == 1);
    assert('0' <= result[0][0] && result[0][0] <= '9');

    auto data = Data{};
    data.label = result[0];

    data.y = Eigen::VectorXd(kDigitsNumber);
    data.y.setConstant(false_score);
    data.y(data.label[0] - '0') = true_score;

    data.x = Eigen::VectorXd(kScanSize);
    for (std::size_t i = 1; i < kColumnsCount; ++i) {
      data.x[i - 1] = std::stod(result[i]) / kShadesCount;
    }

    instances.push_back(std::make_shared<const Data>(std::move(data)));
  }

  return instances;
}

}  // namespace

DataSupplier::DataSupplier(const std::string &train_path,
                           const std::string &test_path,
                           const double false_score, const double true_score) {
  static constexpr std::size_t kTrainInitialSize = 60'000;
  static constexpr std::size_t kValidationSize = 10'000;
  static constexpr std::size_t kTestSize = 10'000;

  spdlog::info("Parsing train data...");
  train_ = ReadMnistCsv(train_path, false_score, true_score);
  assert(train_.size() == kTrainInitialSize);

  validation_ =
      std::vector(std::make_move_iterator(train_.rbegin()),
                  std::make_move_iterator(train_.rbegin() + kValidationSize));
  train_.resize(kTrainInitialSize - kValidationSize);

  spdlog::info("Parsing test data...");
  test_ = ReadMnistCsv(test_path, false_score, true_score);
  assert(test_.size() == kTestSize);
}

std::size_t DataSupplier::GetInputLayerSize() const { return kScanSize; }
std::size_t DataSupplier::GetOutputLayerSize() const { return kDigitsNumber; }

std::vector<std::shared_ptr<const nn::IData>> DataSupplier::GetTrainData()
    const {
  return train_;
}

std::vector<std::shared_ptr<const nn::IData>> DataSupplier::GetValidationData()
    const {
  return validation_;
}

std::vector<std::shared_ptr<const nn::IData>> DataSupplier::GetTestData()
    const {
  return test_;
}

}  // namespace nn
