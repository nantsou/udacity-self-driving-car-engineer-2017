#ifndef MEASUREMENT_PACKAGE_H_
#define MEASUREMENT_PACKAGE_H_

#include "Eigen/Dense"

class MeasurementPackage {
public:
  enum SensorType{
    LASER,
    RADAR
  };

  long long timestamp_;
  SensorType sensor_type_;
  Eigen::VectorXd raw_measurements_;

  MeasurementPackage(const Eigen::VectorXd& rawMeasurement, const SensorType sensorType, long long timeStamp ) {
    raw_measurements_ = rawMeasurement;
    sensor_type_ = sensorType;
    timestamp_ = timeStamp;
  }

  bool isLaser() const { return sensor_type_ == LASER; }
  bool isRadar() const { return sensor_type_ == RADAR; }
  long long getTimeStamp() const { return timestamp_; }
  Eigen::VectorXd getRawMeasurement() const { return raw_measurements_; }
};

#endif /* MEASUREMENT_PACKAGE_H_ */

