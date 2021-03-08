#include <iostream>
#include "ukf.h"
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Define the constants used in ukf.cpp
static const int n_x = 5;
static const int n_aug = 7;
static const int n_sigma = 2 * n_aug + 1;
static const int n_z_radar = 3;
static const int n_z_laser = 2;

// Define private functions

/**
 * Normalize the angles which is greater than pi or less than -pi.
 * @param {double} theta
 */
double UKF::AngleNormalization(const double theta) {
  if (theta > M_PI) {
    return M_PI + fmod(theta + M_PI, 2.*M_PI);
  } else if (theta < -M_PI) {
    return fmod(theta + M_PI, 2.*M_PI) - M_PI;
  } else {
    return theta;
  }
}

/**
 * Prepare the weights of the unscented kalman filter
 */
VectorXd UKF::PrepareWeights() {
  double lambda = 3 - n_aug;
  VectorXd weights = VectorXd(n_sigma);
  weights.fill(0.5/(lambda + n_aug));
  weights(0) = lambda/(lambda + n_aug);
  return weights;
}

/**
 * Transform sigma points to measurement space of lidar (laser)
 * @param {VectorXd} x
 */
VectorXd UKF::LidarTransfrom(const VectorXd &x) {
  return x.head(2);
}

/**
 * Transform sigma points to measurement space of radar
 * @param {VectorXd} x
 */
VectorXd UKF::RadarTransfrom(const VectorXd &x) {
  const double px = x(0);
  const double py = x(1);
  const double v = x(2);
  const double psi = x(3);
  // avoid being divided by 0
  const double eps = 1.0e-6;

  double rho = sqrt(px*px + py*py);
  double phi = atan2(py, px);
  double rho_dot = (px*cos(psi)*v + py*sin(psi)*v) / (eps + rho);

  VectorXd result(3);
  result << rho, phi, rho_dot;
  return result;
}

/**
 * Generate sigma points
 * @param {VectorXd} x, measurement vector
 * @param {MatrixXd} P, covariance matrix
 * @param {VectorXd} process_noise
 */
MatrixXd UKF::SigmaPointGeneration(const VectorXd &x, const MatrixXd &P, const VectorXd process_noise) {
  // define parameter
  const double lambda = 3 - n_aug;

  // define augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug, n_aug);
  P_aug.topLeftCorner(n_x, n_x) = P;
  P_aug(5, 5) = process_noise(0);
  P_aug(6, 6) = process_noise(1);

  // define augmented mean vector
  VectorXd X_aug = VectorXd::Zero(n_aug);
  X_aug.head(n_x) = x;

  // generate sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug, n_sigma);
  const double c = sqrt(lambda + n_aug);
  const MatrixXd A = P_aug.llt().matrixL();

  Xsig_aug.col(0) = X_aug;
  for (int i = 0; i < n_aug; i++) {
    Xsig_aug.col(i + 1) = X_aug + c * A.col(i);
    Xsig_aug.col(i + 1 + n_aug) = X_aug - c * A.col(i);
  }

  return Xsig_aug;
}

/**
 * Predict sigma points per column
 * @param {VectorXd} xsig_aug, augmented sigma vector
 * @param {double} delta_t, time difference between k and k+1
 */
VectorXd UKF::PredictSigmaPoints(const VectorXd &xsig_aug, const double delta_t) {
  // extract the values for the prediction.
  const double v = xsig_aug(2);
  const double psi = xsig_aug(3);
  const double psi_dot = xsig_aug(4);
  const double nu_a = xsig_aug(5);
  const double nu_psi_dot_dot = xsig_aug(6);

  // calculate the values for the prediction.
  const double delta_t2 = delta_t*delta_t;

  // extract x at position k.
  VectorXd x_k = xsig_aug.head(5);

  // declare the container for predicted state and noise
  VectorXd predict_state(5, 1);
  VectorXd noise(5, 1);

  // define the predicted state
  // avoid division by zero
  if (fabs(psi_dot) > 0.001) {
    predict_state << v/psi_dot*(sin(psi + psi_dot*delta_t) - sin(psi)),
                     v/psi_dot*(cos(psi) - cos(psi + psi_dot*delta_t)),
                     0,
                     psi_dot*delta_t,
                     0;
  } else {
    predict_state << v*delta_t*cos(psi),
                     v*delta_t*sin(psi),
                     0,
                     psi_dot*delta_t,
                     0;
  }

  // define noise
  noise << 0.5*nu_a*delta_t2*cos(psi),
           0.5*nu_a*delta_t2*sin(psi),
           nu_a*delta_t,
           0.5*nu_psi_dot_dot*delta_t2,
           nu_psi_dot_dot*delta_t;

  // return the summation of x_k, predict_state and noise as the predicted x.
  return x_k + predict_state + noise;
}

/**
 * Predict sigma points
 * @param {MatrixXd} Xsig_aug, augmented sigma points matrix
 * @param {double} delta_t, time difference between k and k+1
 */
MatrixXd UKF::SigmaPointPrediction(const MatrixXd &Xsig_aug, const double delta_t) {
  MatrixXd Xsig_pred = MatrixXd(n_x, n_sigma);
  for (int i = 0; i < n_sigma; i++) {
    Xsig_pred.col(i) = PredictSigmaPoints(Xsig_aug.col(i), delta_t);
  }
  return Xsig_pred;
}

/**
 * Calculate predicted state mean
 * @param {MatrixXd} Xsig_pred, predicted sigma points matrix
 * @param {VectorXd} weights
 */
VectorXd UKF::MeanPrediction(const MatrixXd &Xsig_pred, const VectorXd &weights) {
  VectorXd x = VectorXd::Zero(Xsig_pred.rows());
  for (int i = 0; i < n_sigma; i++) {
    x += weights(i)*Xsig_pred.col(i);
  }
  return x;
}

/**
 * Calculate predicted state covariance matrix
 * @param {MatrixXd} Xsig, sigma points matrix
 * @param {VectorXd} x, measurement vector
 * @param {VectorXd} weights
 */
MatrixXd UKF::CovariancePrediction(const MatrixXd &Xsig, const VectorXd &x, const VectorXd &weights) {
  MatrixXd P = MatrixXd::Zero(x.rows(), x.rows());
  for (int i = 0; i < n_sigma; i++) {
    VectorXd diff_x = Xsig.col(i) - x;
    P = P + weights(i) * diff_x * diff_x.transpose();
  }
  return P;
}

/**
 * Transform sigma points into measurement space
 * @param {MatrixXd} Xsig_pred, predicted sigma points matrix
 * @param {bool} isLider, check the input maxtrix belongs lidar (laser) or radar
 */
MatrixXd UKF::TransformSigmaPoinstsIntoMeasurementSpace(const MatrixXd &Xsig_pred, const bool isLidar) {

  const int n_rows = isLidar? n_z_laser:n_z_radar;

  MatrixXd Zsig = MatrixXd(n_rows, n_sigma);

  for (int i = 0; i < n_sigma; i++) {
    Zsig.col(i) = isLidar? LidarTransfrom(Xsig_pred.col(i)):RadarTransfrom(Xsig_pred.col(i));
  }

  return Zsig;
}

/**
 * Calculate cross correlation
 * @param {MatrixXd} Xsig, sigma points matrix
 * @param {VectorXd} x, measurement vector
 * @param {MatrixXd} Zsig, measurement-space transformed sigma matrix
 * @param {VectorXd} z, raw measurement vector
 * @param {VectorXd} weights
 * @param {bool} isLider, check the input maxtrix belongs lidar (laser) or radar
 */
MatrixXd UKF::CrossCorrelation(const MatrixXd &Xsig, const VectorXd &x,
                               const MatrixXd &Zsig, const VectorXd &z,
                               const VectorXd &weights) {
  MatrixXd Tc = MatrixXd::Zero(x.rows(), z.rows());

  for (int i = 0; i < n_sigma; i++) {
    VectorXd diff_x = Xsig.col(i) - x;
    VectorXd diff_z = Zsig.col(i) - z;
    diff_x(3) = AngleNormalization(diff_x(3));
    diff_z(1) = AngleNormalization(diff_z(1));
    Tc += weights(i)*diff_x*diff_z.transpose();
  }

  return Tc;
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF():
is_initialized_(false),
use_laser_(true),
use_radar_(true) {
  
  // Initial state vector
  x_ = VectorXd(5);
  
  // Initial covariance matrix
  P_ = MatrixXd::Zero(n_x, n_x);
  for (int i = 0; i < n_x; ++i) {
    P_(i,i) = 0.05;
  }
  
  // Define weights
  weights_ = PrepareWeights();
  
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;
  
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/3;
  
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;
  
  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;
  
  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;
  
  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;
  
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  // Define measurement noise covariance matrix of radar
  R_radar_ = MatrixXd::Zero(3, 3);
  R_radar_(0, 0) = pow(std_radr_, 2);
  R_radar_(1, 1) = pow(std_radphi_, 2);
  R_radar_(2, 2) = pow(std_radrd_, 2);
  
  // Define measurement noise covariance matrix of laser
  R_lidar_ = MatrixXd::Zero(2, 2);
  R_lidar_(0, 0) = pow(std_laspx_, 2);
  R_lidar_(1, 1) = pow(std_laspy_, 2);
  
  // Define process noise
  process_noise_ = VectorXd::Zero(2);
  process_noise_ << pow(std_a_, 2), pow(std_yawdd_, 2);
}

UKF::~UKF() {}

/**
 * Initialize
 * @param {MeasurementPackage} meas_package The first measurement data of
 * either radar or laser
 */
void UKF::Initialize(const MeasurementPackage &meas_package) {

  double px = 0.0;
  double py = 0.0;
  double v = 0.0;
  double psi = 0.0;
  double psi_dot = 0.0;

  // get first measurement info
  const auto &raw_measurement = meas_package.getRawMeasurement();

  if (meas_package.isLaser()) {

    px = raw_measurement(0);
    py = raw_measurement(1);

  } else if (meas_package.isRadar()) {

    double rho = raw_measurement(0);
    double phi = raw_measurement(1);
    px = rho*cos(phi);
    py = rho*sin(phi);

    v = raw_measurement(2);
  }

  x_ << px, py, v, psi, psi_dot;
  previous_timestamp_ = meas_package.timestamp_;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage &meas_package) {
  if (!is_initialized_) {
    Initialize(meas_package);
    is_initialized_ = true;
    return;
  }

  double delta_t = (meas_package.timestamp_ - previous_timestamp_)/1.0e6;

  // divide time step into smaller one to increase the numerical stability
  while (delta_t > 0.1) {
    Predict(0.05);
    delta_t -= 0.05;
  }


  Predict(delta_t);
  previous_timestamp_ = meas_package.timestamp_;

  if (meas_package.isLaser() && use_laser_) {
    UpdateLidar(meas_package);
  }

  if (meas_package.isRadar() && use_radar_ ) {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Predict(double delta_t) {
  const MatrixXd Xsig = SigmaPointGeneration(x_, P_, process_noise_);

  Xsig_pred_ = SigmaPointPrediction(Xsig, delta_t);

  x_ = MeanPrediction(Xsig_pred_, weights_);
  x_(3) = AngleNormalization(x_(3));
  P_ = CovariancePrediction(Xsig_pred_, x_, weights_);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage &meas_package) {
  const MatrixXd Zsig = TransformSigmaPoinstsIntoMeasurementSpace(Xsig_pred_, meas_package.isLaser());
  const MatrixXd z = meas_package.getRawMeasurement();
  const MatrixXd Tc = CrossCorrelation(Xsig_pred_, x_, Zsig, z, weights_);

  // predict measurement and covariance
  const VectorXd z_pred = MeanPrediction(Zsig, weights_);
  const MatrixXd S = CovariancePrediction(Zsig, z_pred, weights_) + R_lidar_;

  const MatrixXd K = Tc*S.inverse();

  VectorXd diff_z = z - z_pred;

  x_ = x_ + K*diff_z;
  x_(3) = AngleNormalization(x_(3));
  P_ = P_ - K*S*K.transpose();

  NIS_laser_ = diff_z.transpose() * S.inverse() * diff_z;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage &meas_package) {
  const MatrixXd Zsig = TransformSigmaPoinstsIntoMeasurementSpace(Xsig_pred_, meas_package.isLaser());
  const MatrixXd z = meas_package.getRawMeasurement();
  const MatrixXd Tc = CrossCorrelation(Xsig_pred_, x_, Zsig, z, weights_);

  // predict measurement and covariance
  const VectorXd z_pred = MeanPrediction(Zsig, weights_);
  const MatrixXd S = CovariancePrediction(Zsig, z_pred, weights_) + R_radar_;

  const MatrixXd K = Tc*S.inverse();
  VectorXd diff_z = z - z_pred;
  diff_z(1)=AngleNormalization(diff_z(1));

  x_ = x_ + K*diff_z;
  x_(3) = AngleNormalization(x_(3));
  P_ = P_ - K*S*K.transpose();

  NIS_radar_ = diff_z.transpose() * S.inverse() * diff_z;
}