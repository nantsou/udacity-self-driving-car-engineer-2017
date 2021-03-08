#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"

class FusionEKF {
public:

  // Constructor.
  FusionEKF();

  // Destructor.
  virtual ~FusionEKF();

  // Run the whole flow of the Kalman Filter from here.
  // @param measurement_pack The measurement pack at k+1
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  // Kalman Filter update and prediction math lives in here.
  KalmanFilter ekf_;

private:
  // Check whether the kalman filter is initialized or not.
  bool is_initialized_;

  // Record the time stamp of previous point.
  long long previous_timestamp_;

  // Measurement covariance matrix - radar
  MatrixXd R_radar_;

  // Jacobian Measurement function - radar
  MatrixXd Hj_;

  // Measurement covariance matrix - laser
  MatrixXd R_laser_;

  // Measurement function matrix - laser
  MatrixXd H_laser_;

  // Initializing Kalman filter within FusionEKF
  // @param measurement_pack The measurement pack of first point
  void Initialize(const MeasurementPackage &measurement_pack);

  // Predict the measurement at k+1 within FusionEKF
  void Predict(const MeasurementPackage &measurement_pack);

  // Update the state at k within FusionEKF
  void Update(const MeasurementPackage &measurement_pack);

  // Utility used to calculate predict measurement for radar record
  VectorXd RaderPredictMeasurement(VectorXd &x) const;
};

#endif /* FusionEKF_H_ */
