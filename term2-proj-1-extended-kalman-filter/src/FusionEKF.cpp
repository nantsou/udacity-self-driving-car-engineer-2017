#include "FusionEKF.h"
#include "tools.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

FusionEKF::FusionEKF():
  is_initialized_(false),
  previous_timestamp_(0),
  R_radar_(MatrixXd(3, 3)),
  Hj_(MatrixXd(3, 4)),
  R_laser_(MatrixXd(2, 2)),
  H_laser_(MatrixXd(2, 4))
{

  // set up R_radar with the values suggested by the project.
  R_radar_ << 0.09, 0.0, 0.0,
              0.0, 0.0009, 0.0,
              0.0, 0.0, 0.09;

  // set up R_raser with the values suggested by the project.
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // set up H_raser with the values suggested by the project.
  H_laser_ << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0;
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  if (!is_initialized_) {
    Initialize(measurement_pack);
    is_initialized_ = true;
    return;
  }

  Predict(measurement_pack);

  Update(measurement_pack);
}

void FusionEKF::Initialize(const MeasurementPackage &measurement_pack) {

  // Define the container of px, py, vx and vy, for initialization.
  VectorXd x = VectorXd(4);

  // Define the container of P, covariance matrix, for initialization.
  MatrixXd P = MatrixXd(4,4);
  P << 1.0, 0.0, 0.0, 0.0,
       0.0, 1.0, 0.0, 0.0,
       0.0, 0.0, 1000.0, 0.0,
       0.0, 0.0, 0.0, 1000.0;

  // Declare the container of Q, process covariance matrix, for initialization.
  MatrixXd Q = MatrixXd(4,4);

  // Define the container of F, state transition matrix, for initialization
  MatrixXd F = MatrixXd(4,4);
  F << 1.0, 0.0, 1.0, 0.0,
       0.0, 1.0, 0.0, 1.0,
       0.0, 0.0, 1.0, 0.0,
       0.0, 0.0, 0.0, 1.0;

  // store the first timestamp as the beginning.
  previous_timestamp_ = measurement_pack.timestamp_;

  // calculate the initial values for px, py, vx and vy.
  double px = 0.0;
  double py = 0.0;
  double vx = 0.0;
  double vy = 0.0;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // convert rho, phi and rho_dot to px, py, vx and vy if the first record if RADAR.
    double rho = measurement_pack.raw_measurements_[0];
    double phi = measurement_pack.raw_measurements_[1];
    double rho_dot = measurement_pack.raw_measurements_[2];

    px = rho * cos(phi);
    py = rho * sin(phi);
    vx = rho_dot * cos(phi);
    vy = rho_dot * sin(phi);

  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    // set px and py and keep vx and vy eqaul to 0 if the record is LASER.
    px = measurement_pack.raw_measurements_[0];
    py = measurement_pack.raw_measurements_[1];
  }

  // set the calculated px, py, vx and vy to x for initialization.
  x << px, py, vx, vy;

  // initialize extended kalman filter with x, P, F and Q.
  ekf_.Init(x, P, F, Q);
}

void FusionEKF::Predict(const MeasurementPackage &measurement_pack) {
  // calculate delta time
  const long long current_time_stamp = measurement_pack.timestamp_;
  const double dt = (current_time_stamp - previous_timestamp_)/1.0e6;

  // store the current time stamp for next point.
  previous_timestamp_ = current_time_stamp;

  // modify the F matrix so that the time is integrated
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  // set the process covariance matrix Q
  double dt_2 = dt*dt;
  double dt_3 = dt_2*dt;
  double dt_4 = dt_3*dt;
  // set process noise variances of x and y to 9 which is suggested by the project.
  const double process_noise = 9;

  ekf_.Q_ << dt_4 / 4 * process_noise, 0, dt_3 / 2 * process_noise, 0,
          0, dt_4 / 4 * process_noise, 0, dt_3 / 2 * process_noise,
          dt_3 / 2 * process_noise, 0, dt_2 * process_noise, 0,
          0, dt_3 / 2 * process_noise, 0, dt_2 * process_noise;

  ekf_.Predict();
}

void FusionEKF::Update(const MeasurementPackage &measurement_pack) {
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    const VectorXd z = measurement_pack.raw_measurements_;
    const VectorXd z_pred = RaderPredictMeasurement(ekf_.x_);
    Hj_ = tools::CalculateJacobian(ekf_.x_);
    ekf_.PostUpdate(z, z_pred, Hj_, R_radar_);
  } else {
    // Laser updates
    ekf_.Update(measurement_pack.raw_measurements_, H_laser_, R_laser_);
  }
}

VectorXd FusionEKF::RaderPredictMeasurement(VectorXd &x) const {
  const double px = x(0);
  const double py = x(1);
  const double vx = x(2);
  const double vy = x(3);
  // avoid being divided by 0
  const double eps = 1.0e-6;

  const double rho = sqrt(px*px + py*py);
  const double phi = atan2(py, px);
  const double rho_dot = (px * vx + py * vy) / (eps + rho);

  VectorXd result(3);
  result << rho, phi, rho_dot;
  return result;
}