//
// Created by Joey Liu on 2017/06/17.
//
#include <math.h>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"

#ifndef MPC_UTILS_H
#define MPC_UTILS_H

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
static double deg2rad(double x) { return x * pi() / 180; }
static double rad2deg(double x) { return x * 180 / pi(); }

// Evaluate a polynomial.
static double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

static vector<Eigen::VectorXd> trandformCoordinates(const double px, const double py, const double psi,
                                             const vector<double> &ptsx, const vector<double> &ptsy) {
  // Define containers for vehicle base coordinates
  Eigen::VectorXd ptsx_vehicle(ptsx.size());
  Eigen::VectorXd ptsy_vehicle(ptsy.size());

  // Transform the coordinates to vehicle base coordinates
  for (int i = 0; i < ptsx.size(); i++) {
    const double dx = ptsx[i] - px;
    const double dy = ptsy[i] - py;
    ptsx_vehicle[i] = dx * cos(psi) + dy * sin(psi);
    ptsy_vehicle[i] = dy * cos(psi) - dx * sin(psi);
  }
  return {ptsx_vehicle, ptsy_vehicle};
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
static Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

#endif //MPC_UTILS_H
