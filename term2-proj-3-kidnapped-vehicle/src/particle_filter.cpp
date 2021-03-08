/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // set the number of particles
  num_particles = 500;

  // initialize the particles
  // create random number generator
  random_device rd;
  mt19937 gen(rd());
  // create gaussian distributions for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // set particles weights to 1
  weights.clear();
  weights.resize((unsigned long)num_particles, 1.0);

  // initialize particles' container
  particles.clear();
  particles.resize((unsigned long)num_particles);

  for (auto &particle:particles) {
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // get standard deviations for x, y and theta
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // get measurement yaw
  double m_yaw = yaw_rate*delta_t;

  // create random number generator
  random_device rd;
  mt19937 gen(rd());

  // create gaussian distributions for noise
  normal_distribution<double> dist_noise_x(0, std_x);
  normal_distribution<double> dist_noise_y(0, std_y);
  normal_distribution<double> dist_noise_theta(0, std_theta);

  // define the variables used for prediction
  double yaw, new_x, new_y, new_theta;

  for (auto &particle:particles) {
    yaw = particle.theta;
    if (yaw_rate != 0.0) {
      new_x = particle.x + velocity/yaw_rate*(sin(yaw + m_yaw) - sin(yaw));
      new_y = particle.y + velocity/yaw_rate*(cos(yaw) - cos(yaw + m_yaw));
      new_theta = particle.theta + m_yaw;
    } else {
      new_x = particle.x + velocity*delta_t*cos(yaw);
      new_y = particle.y + velocity*delta_t*sin(yaw);
      new_theta = particle.theta;
    }

    // predict particles
    particle.x = new_x + dist_noise_x(gen);
    particle.y = new_y + dist_noise_y(gen);
    particle.theta = new_theta + dist_noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for (auto const& prediction:predicted) {
    double min_dist = numeric_limits<double>::max();
    for (auto &observation:observations) {
      // skip the observations which are already assigned
      if (observation.id > 0) continue;
      // get current distance between observation and prediction
      double current_dist = dist(observation.x, observation.y, prediction.x, prediction.y);
      // assign the observation if current distance smaller than previous minimum distance
      if (current_dist < min_dist) {
        observation.id = prediction.id;
        min_dist = current_dist;
      }
    }
  }

  // remove the observations which are not the closest ones.
  observations.erase(
    remove_if(
            observations.begin(),
            observations.end(),
            // remove the observation which id is 0 or -1
            [] (const LandmarkObs& obs_lm) {return obs_lm.id < 1;}),
    observations.end()
  );
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  // set sigma for using later
  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];

  // calculate coefficients independent of particle, landmark, or observation
  // for multivariate gaussian probability density.
  double denominator = 2.0*M_PI*sigma_x*sigma_y;
  double denominator_x = pow(sigma_x, 2);
  double denominator_y = pow(sigma_y, 2);


  // update particles
  for (auto &particle:particles) {
    // convert observations into map spatial coordinates
    vector<LandmarkObs> observations_map(observations);
    for (auto &observation:observations_map) {
      double o_map_x = observation.x*cos(particle.theta) - observation.y*sin(particle.theta) + particle.x;
      double o_map_y = observation.x*sin(particle.theta) + observation.y*cos(particle.theta) + particle.y;
      observation.x = o_map_x;
      observation.y = o_map_y;
      observation.id = 0;
    }

    // create a container for land mark predictions
    vector<LandmarkObs> predictions;

    for (auto const& landmark:map_landmarks.landmark_list) {
      // collect the landmark as prediction which distance within the sensor range
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) < sensor_range ) {
        LandmarkObs prediction;
        prediction = {landmark.id_i, landmark.x_f, landmark.y_f};
        predictions.push_back(prediction);
      }
    }

    // perform data association
    dataAssociation(predictions, observations_map);

    // create a mapper for looking up predicted landmarks
    map<int, LandmarkObs> predicted_landmark_mapper;
    for (auto const& prediction:predictions) {
      predicted_landmark_mapper.insert({prediction.id, prediction});
    }

    // re-initialize weight
    double weight = 1.0;

    for (auto const& observation:observations_map) {
      LandmarkObs prediction = predicted_landmark_mapper[observation.id];

      // calculate multivariate gaussian probability density for updating weights
      double c_x = pow(observation.x - prediction.x, 2)/denominator_x;
      double c_y = pow(observation.y - prediction.y, 2)/denominator_y;
      double measurement_weight = exp(-0.5*(c_x + c_y))/denominator;

      // replace weights, which are smaller than 0.001, with 0.001 for numerical stability
      if (measurement_weight < 1.0e-3) {measurement_weight = 1.0e-3;}
      weight *= measurement_weight;
    }
    particle.weight = weight;
  }
  // update particle_filter's weights
  for (int i = 0; i < num_particles; i++) {
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  // create the random generator;
  random_device rd;
  mt19937 gen(rd());
  // create discrete_distribution with weights
  discrete_distribution<> dist_particles(weights.begin(), weights.end());
  // create the container for resampled particles
  vector<Particle> resampled_particles((unsigned long) num_particles);
  // resample the particles
  for (int i = 0; i < num_particles; i++) {
    resampled_particles[i] = particles[dist_particles(gen)];
  }
  // replace particle_filter's particles with resampled particles
  particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
