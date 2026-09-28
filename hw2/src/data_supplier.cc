#include "data_supplier.h"

#include <iterator>
#include <stdexcept>

#include "mnist_csv.h"

namespace hw2 {

DataSupplier::DataSupplier(const std::string& train_path,
                           const std::string& test_path, double false_score,
                           double true_score, std::size_t validation_size) {
  training_ = nn::ReadMnistCsv<Data>(train_path, false_score, true_score);
  if (validation_size == 0 || training_.size() <= validation_size) {
    throw std::invalid_argument(
        "MNIST training data must exceed the nonzero validation size");
  }
  const auto split =
      training_.end() - static_cast<std::ptrdiff_t>(validation_size);
  validation_.assign(std::make_move_iterator(split),
                     std::make_move_iterator(training_.end()));
  training_.erase(split, training_.end());
  testing_ = nn::ReadMnistCsv<Data>(test_path, false_score, true_score);
}

}  // namespace hw2
