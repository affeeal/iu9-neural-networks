#include <iostream>

#include "data_supplier.h"

int main(int argc, char* argv[]) {
  const auto data_supplier = lab1::DataSupplier(0.0, 1.0);

  const auto training = data_supplier.GetTrainingData();
  std::cout << training.size() << '\n';

  const auto validation = data_supplier.GetValidationData();
  std::cout << validation.size() << '\n';

  const auto testing = data_supplier.GetTestingData();
  std::cout << testing.size() << '\n';
}
