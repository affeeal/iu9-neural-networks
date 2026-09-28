#pragma once

#include "activation_function.h"
#include "cost_function.h"

namespace nn {

inline bool IsCategorical(ICostFunction& cost) {
  return dynamic_cast<CrossEntropy*>(&cost) ||
         dynamic_cast<KLDivergence*>(&cost);
}

// Fuse softmax with categorical losses: avoid 0 * infinity after underflow.
inline Eigen::VectorXd OutputDelta(IActivationFunction& activation,
                                   ICostFunction& cost,
                                   const Eigen::VectorXd& y,
                                   const Eigen::VectorXd& z,
                                   const Eigen::VectorXd& a) {
  if (dynamic_cast<Softmax*>(&activation) && IsCategorical(cost)) {
    return a * y.sum() - y;
  }
  return activation.Jacobian(z).transpose() * cost.GradientWrtActivations(y, a);
}

inline double OutputCost(IActivationFunction& activation, ICostFunction& cost,
                         const Eigen::VectorXd& y, const Eigen::VectorXd& z) {
  if (dynamic_cast<Softmax*>(&activation) && IsCategorical(cost)) {
    const Eigen::ArrayXd shifted = z.array() - z.maxCoeff();
    const double log_sum = std::log(shifted.exp().sum());
    double result = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
      if (y[i] != 0) {
        result += y[i] * (log_sum - shifted[i]);
        if (dynamic_cast<KLDivergence*>(&cost)) result += y[i] * std::log(y[i]);
      }
    }
    return result;
  }
  return cost.Apply(y, activation.Apply(z));
}

}  // namespace nn
