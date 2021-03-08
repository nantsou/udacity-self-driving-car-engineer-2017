#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

#include "Eigen/Dense"
#include "measurement_package.h"
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
private:
  /**
   * Normalize the angles which is greater than pi or less than -pi.
   * @param theta
   */
  double AngleNormalization(const double phi);

  /**
   * Prepare the weights of the unscented kalman filter
   */
  VectorXd PrepareWeights();

  /**
   * Transform sigma points to measurement space of lidar (laser)
   * @param x
   */
  VectorXd LidarTransfrom(const VectorXd &x);

  /**
   * Transform sigma points to measurement space of radar
   * @param x
   */
  VectorXd RadarTransfrom(const VectorXd &x);

  /**
   * Generate sigma points
   * @param x, measurement vector
   * @param P, covariance matrix
   * @param process_noise
   */
  MatrixXd SigmaPointGeneration(const VectorXd &x, const MatrixXd &P, const VectorXd process_noise);

  /**
   * Predict sigma points per column
   * @param xsig_aug, augmented sigma vector
   * @param delta_t, time difference between k and k+1
   */
  VectorXd PredictSigmaPoints(const VectorXd &x_pred, const double delta_t);

  /**
   * Predict sigma points
   * @param Xsig_aug, augmented sigma points matrix
   * @param delta_t, time difference between k and k+1
   */
  MatrixXd SigmaPointPrediction(const MatrixXd &Xsig_aug, const double delta_t);

  /**
   * Calculate predicted state mean
   * @param Xsig_pred, predicted sigma points matrix
   * @param weights
   */
  VectorXd MeanPrediction(const MatrixXd &Xsig_pred, const VectorXd &weights);

  /**
   * Calculate predicted state covariance matrix
   * @param Xsig, sigma points matrix
   * @param x, measurement vector
   * @param weights
   */
  MatrixXd CovariancePrediction(const MatrixXd &Xsig_pred, const VectorXd &x, const VectorXd &weights);

  /**
   * Transform sigma points into measurement space
   * @param Xsig_pred, predicted sigma points matrix
   * @param isLider, check the input maxtrix belongs lidar (laser) or radar
   */
  MatrixXd TransformSigmaPoinstsIntoMeasurementSpace(const MatrixXd &Xsig_pred, const bool isLidar);

  /**
   * Calculate cross correlation
   * @param Xsig, sigma points matrix
   * @param x, measurement vector
   * @param Zsig, measurement-space transformed sigma matrix
   * @param z, raw measurement vector
   * @param weights
   * @param isLider, check the input maxtrix belongs lidar (laser) or radar
   */
  MatrixXd CrossCorrelation(const MatrixXd &Xsig, const VectorXd &x,
                                const MatrixXd &Zsig, const VectorXd &z,
                                const VectorXd &weights);

public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* record the timestamp
  long long previous_timestamp_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Process noise vector for accelerations:
  Eigen::VectorXd process_noise_;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;

  ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

  ///* radar measurement covariance matrix
  Eigen::MatrixXd R_radar_;

  ///* laser measurement covariance matrix
  Eigen::MatrixXd R_lidar_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * Initialize
   * @param meas_package The first measurement data of either radar or laser
   */
  void Initialize(const MeasurementPackage &meas_package);

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(const MeasurementPackage &meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Predict(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(const MeasurementPackage &meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(const MeasurementPackage &meas_package);
};

#endif /* UKF_H */
