#include <iostream>
#include "tools.h"

namespace tools {
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                         const vector<VectorXd> &ground_truth) {

    // define the container of rmse, root mean square error.
    VectorXd rmse(4);
    rmse << 0.0, 0.0, 0.0, 0.0;

    // if the sizes of estimations and ground_truth are different
    // or the size of estimations is 0.
    // return 0.0, 0.0, 0.0, 0.0 as the rmse result.
    if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
      cout << "Invalid estimation or ground_truth data" << endl;
      return rmse;
    }

    //accumulate squared residuals
    for (int i = 0; i < estimations.size(); i++) {
      VectorXd residual = estimations[i] - ground_truth[i];
      residual = residual.array() * residual.array();
      rmse += residual;
    }

    //calculate the mean
    rmse = rmse/estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
  }

  MatrixXd CalculateJacobian(const VectorXd &x_state) {
    MatrixXd Hj(3, 4);

    // recover state parameters
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    // avoid being divided by 0.
    const double eps = 1.0e-6;

    // get coefficients for calculating Jacobain
    double c1 = std::max(eps, px*px + py*py);
    double c2 = std::max(eps, sqrt(c1));
    double c3 = std::max(eps, (c1 * c2));

    // calculate Jacobain
    Hj << (px/c2), (py/c2), 0, 0,
            -(py/c1), (px/c1), 0, 0,
            py*(vx*py - vy*px)/c3, px*(vy*px - vx*py)/c3, (px/c2), (py/c2);

    return Hj;
  }
}