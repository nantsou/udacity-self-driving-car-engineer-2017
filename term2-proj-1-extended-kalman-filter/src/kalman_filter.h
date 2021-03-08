#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class KalmanFilter {
public:

  // State vector
  VectorXd x_;

  // State covariance matrix
  MatrixXd P_;

  // State transistion matrix
  MatrixXd F_;

  // Process covariance matrix
  MatrixXd Q_;

  // Identity matrix
  MatrixXd I_;

  // Constructor
  KalmanFilter();

  // Destructor
  virtual ~KalmanFilter();

  /**
  * Init Initializes Kalman filter
  * @param x_in Initial state
  * @param P_in Initial state covariance
  * @param F_in Transition matrix
  * @param Q_in Process covariance matrix
  */
  void Init(VectorXd &x_in, MatrixXd  &P_in, MatrixXd &F_in, MatrixXd &Q_in);

  /**
  * Prediction Predicts the state and the state covariance
  * using the process model
  * @param delta_T Time between k and k+1 in s
  */
  void Predict();

  /**
  * Updates the state by using PostUpdate
  * In this function, only predict z is calcuated
  * @param z The measurement at k+1
  * @param H_in Measurement matrix
  * @param R_in Measurement covariance matrix
  */
  void Update(const VectorXd &z, const MatrixXd &H, const MatrixXd &R);

  /**
  * Updates the state with calculated predict z by using Extended Kalman Filter equations
  * @param z The measurement at k+1
  * @param z_pred The predicted measurement at k
  * @param H_in Measurement matrix
  * @param R_in Measurement covariance matrix
  */
  void PostUpdate(const VectorXd &z, const VectorXd &z_pred, const MatrixXd &H, const MatrixXd &R);

};

#endif /* KALMAN_FILTER_H_ */
