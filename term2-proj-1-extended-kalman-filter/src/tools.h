#ifndef TOOLS_H_
#define TOOLS_H_

#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

namespace tools {
  // A helper method to calculate RMSE.
  VectorXd CalculateRMSE(const std::vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  // A helper method to calculate Jacobians.
  MatrixXd CalculateJacobian(const VectorXd& x_state);

};

#endif /* TOOLS_H_ */
