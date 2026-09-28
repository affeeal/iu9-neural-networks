#include "data_supplier.h"

#include <iterator>
#include <stdexcept>

#include "mnist_csv.h"

namespace nn {

DataSupplier::DataSupplier(const std::string& train_path,
                           const std::string& test_path, double false_score,
                           double true_score, std::size_t validation_size) {
  train_ = nn::ReadMnistCsv<Data>(train_path, false_score, true_score);
  if (validation_size == 0 || train_.size() <= validation_size) {
    throw std::invalid_argument(
        "MNIST training data must exceed the nonzero validation size");
  }
  const auto split =
      train_.end() - static_cast<std::ptrdiff_t>(validation_size);
  validation_.assign(std::make_move_iterator(split),
                     std::make_move_iterator(train_.end()));
  train_.erase(split, train_.end());
  test_ = nn::ReadMnistCsv<Data>(test_path, false_score, true_score);
}
std::size_t DataSupplier::GetInputLayerSize() const { return 784; }
std::size_t DataSupplier::GetOutputLayerSize() const { return 10; }

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
