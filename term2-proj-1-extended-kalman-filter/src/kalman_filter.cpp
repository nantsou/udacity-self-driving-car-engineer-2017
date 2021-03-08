#include "kalman_filter.h"

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  Q_ = Q_in;

  // Define the Identity matrix with the size of P.
  I_ = MatrixXd::Identity(P_.rows(), P_.cols());
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z, const MatrixXd &H, const MatrixXd &R) {
  const VectorXd z_pred = H * x_;
  PostUpdate(z, z_pred, H, R);
}

void KalmanFilter::PostUpdate(const VectorXd &z, const VectorXd &z_pred, const MatrixXd &H, const MatrixXd &R) {
  VectorXd y =  z - z_pred;
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd K = P_ * Ht * S.inverse();

  // new estimate
  x_ = x_ + (K * y);
  P_ = (I_ - K * H) * P_;
}