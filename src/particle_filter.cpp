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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    if (!is_initialized) {
        default_random_engine random_generator;

        const double& std_x = std[0];
        const double& std_y = std[1];
        const double& std_theta = std[2];

        // Number of particles in the filter
        num_particles = 500;

        // Create normal distributions for x, y and theta
        normal_distribution<double> dist_x(x, std_x);
        normal_distribution<double> dist_y(y, std_y);
        normal_distribution<double> dist_theta(theta, std_theta);

        // Clear the current set of particles and the weights
        particles.clear();
        weights.clear();

        // Create <num_particles> particles, sampled from the normal distributions
        for (int i = 0; i < num_particles; ++i) {
            Particle particle;
            particle.id = i;
            particle.x = dist_x(random_generator);
            particle.y = dist_y(random_generator);
            particle.theta = dist_theta(random_generator);
            particle.weight = 1.0;
            particles.push_back(particle);
            weights.push_back(particle.weight);
        }

        is_initialized = true;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    // TODO make sure that std_pos has exactly these two values!

    default_random_engine random_generator;
    normal_distribution<double> dist_velocity(velocity, std_pos[0]);
    normal_distribution<double> dist_yaw_rate(yaw_rate, std_pos[1]);

    // TODO isn't this a misunderstanding? do we really change the prediction result here?

    const double noisy_velocity = dist_velocity(random_generator);
    const double noisy_yaw_rate = dist_yaw_rate(random_generator);

    // TODO watch the case when yaw rate is zero!

    for (unsigned int i = 0; i < particles.size(); ++i) {
        const double& yaw = particles[i].theta;
        const double yaw_rate_dt = noisy_yaw_rate * delta_t;
        const double v_over_yawr = noisy_velocity/noisy_yaw_rate;
        particles[i].x += (v_over_yawr*(sin(yaw + yaw_rate_dt)-sin(yaw)));
        particles[i].y += (v_over_yawr*(cos(yaw)-cos(yaw + yaw_rate_dt)));
        particles[i].theta += yaw_rate_dt;
    }

	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

    for (unsigned int i = 0; i < particles.size(); ++i) {

        const double& xP = particles[i].x;
        const double& yP = particles[i].y;
        const double& thP = particles[i].theta;
        const double& cs = cos(thP);
        const double& sn = sin(thP);
        const double& stdx = std_landmark[0];
        const double& stdy = std_landmark[1];

        std::vector<int> associations;
        std::vector<double> sense_x;
        std::vector<double> sense_y;
        for (unsigned int o = 0; o < observations.size(); ++o) {
            const double& x = observations[o].x;
            const double& y = observations[o].y;

            // calculate the world coordinates of the measurement
            const double xW = x * cs - y * sn + xP;
            const double yW = x * sn + y * cs + yP;

            // find closest landmark
            double min_distance = std::numeric_limits<double>::max();
            int best_id = 0;
            for (unsigned int l = 0; l < map_landmarks.landmark_list.size(); ++l) {
                const double& xLM = map_landmarks.landmark_list[l].x_f;
                const double& yLM = map_landmarks.landmark_list[l].y_f;
                const double distance = dist(xW, yW, xLM, yLM);
                if (distance < min_distance) {
                    best_id = map_landmarks.landmark_list[l].id_i;
                    min_distance = distance;
                }
            }

            // save the associations for reference
            associations.push_back(best_id);
            sense_x.push_back(xW);
            sense_y.push_back(yW);
        }

        SetAssociations(particles[i], associations, sense_x, sense_y);

        // calculate the product of the multivariate gaussians
        for (unsigned int a = 0; a < associations.size(); ++a) {
            const double& xLM = map_landmarks.landmark_list[associations[a]].x_f;
            const double& yLM = map_landmarks.landmark_list[associations[a]].y_f;
            const double& xW = sense_x[a];
            const double& yW = sense_y[a];

            //particles[i].weight *= 1.0/(2.0*M_PI*stdx*stdy)*exp(-(((xLM-xW)*(xLM-xW)/(2*stdx*stdx)) + ((yLM-yW)*(yLM-yW)/(2*stdy*stdy))));
        }
    }

	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
