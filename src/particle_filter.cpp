/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Edited on: Feb 10, 2020
 * Edited by: Josh White 
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  // Set the number of particles
  num_particles = 100;  
  
  // Create normal distributions for x, y, and theta
  // Adding noise to particles
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  std::default_random_engine gen;
  
  // Resize the particle list
  particles.resize(num_particles);
  
  for (auto& p : particles){ // Props to Junsheng Fu for inadvertantly teaching me about range-based for loops
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    }
  
  // To pass if statement in main.cpp
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  std::default_random_engine gen;
  
  // Generate random Gaussian noise
  std::normal_distribution<double> N_x(0, std_pos[0]);
  std::normal_distribution<double> N_y(0, std_pos[1]);
  std::normal_distribution<double> N_theta(0, std_pos[2]);
  
  for (auto& p : particles){

    // add measurements to each particle
    if (fabs(yaw_rate) < 0.0001){  // if rate of change (velocity) is constant
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);

    } else { // rate of change is variable
      p.x += velocity / yaw_rate * ( sin( p.theta + yaw_rate*delta_t ) - sin(p.theta) );
      p.y += velocity / yaw_rate * ( cos( p.theta ) - cos( p.theta + yaw_rate*delta_t ) );
      p.theta += yaw_rate * delta_t;
    }

    // Final predicted particles with the Gaussian sensor noise
    p.x += N_x(gen);
    p.y += N_y(gen);
    p.theta += N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (auto& obs : observations){
    double minD = std::numeric_limits<float>::max(); //minimum distance

    for (const auto& pred : predicted){
      double distance = dist(obs.x, obs.y, pred.x, pred.y);
      if (minD > distance){
        minD = distance;
        obs.id = pred.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  for (auto& p : particles){
    p.weight = 1.0;

    // Collect only valid landmarks
    vector<LandmarkObs> predictions;
    for(const auto& lm: map_landmarks.landmark_list){
      double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
      if( distance < sensor_range){ // save to predictions if the landmark is within the sensor range
        predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
    }
    
    // Translate 'observations' coordinates from vehicle to map
    vector<LandmarkObs> observations_map;
    double cos_theta = cos(p.theta);
    double sin_theta = sin(p.theta);

    for (const auto& obs : observations){
      LandmarkObs tmp;
      tmp.x = obs.x * cos_theta - obs.y * sin_theta + p.x;
      tmp.y = obs.x * sin_theta + obs.y * cos_theta + p.y;
      observations_map.push_back(tmp);
    }
    
    // Find landmark index for each observation
    dataAssociation(predictions, observations_map); // using final vectors from Steps 1 & 2
    
    // step 4: compute the particle's weight:
    // see equation this link:
    for (const auto& obs_m : observations_map){

      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);
      double x_term = pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
      double y_term = pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      p.weight *=  w;
    }

    weights.push_back(p.weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  num_particles = 100;
  
  // Generate distribution according to weights
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  // Create resampled particles
  vector<Particle> resampled_particles;
  resampled_particles.resize(num_particles);

  // Resample the particles according to weights
  for (int i = 0; i < num_particles; i++){
    int idx = dist(gen);
    resampled_particles[i] = particles[idx];
  }

  // Assign the resampled_particles to the previous particles
  particles = resampled_particles;

  // Clear the weight vector for the next round
  weights.clear();

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}