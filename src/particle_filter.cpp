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

namespace {
    static double normalize(const double angle_rad) {
        double TWO_PI = 2*M_PI;
        return angle_rad - TWO_PI * floor((angle_rad + M_PI) / TWO_PI );
    }
}

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

    default_random_engine random_generator;
    normal_distribution<double> dist_velocity(velocity, std_pos[0]);
    normal_distribution<double> dist_yaw_rate(yaw_rate, std_pos[1]);

    for (unsigned int i = 0; i < particles.size(); ++i) {
        const double noisy_velocity = dist_velocity(random_generator);
        const double noisy_yaw_rate = dist_yaw_rate(random_generator);
        const double& yaw = particles[i].theta;
        const double yaw_rate_dt = noisy_yaw_rate * delta_t;

        if (yaw_rate > 0.001) {
            const double v_over_yawr = noisy_velocity/noisy_yaw_rate;
            particles[i].x += (v_over_yawr*(sin(yaw + yaw_rate_dt)-sin(yaw)));
            particles[i].y += (v_over_yawr*(cos(yaw)-cos(yaw + yaw_rate_dt)));
        }
        else {
            const double velocity_dt = noisy_velocity * delta_t;
            particles[i].x += velocity_dt * cos(yaw);
            particles[i].y += velocity_dt * sin(yaw);
        }

        particles[i].theta += yaw_rate_dt;
        particles[i].theta = normalize(particles[i].theta);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

    double sum_of_weights = 0.0;
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

            // save the associations
            associations.push_back(best_id);
            sense_x.push_back(xW);
            sense_y.push_back(yW);
        }

        particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);

        // calculate the product of the multivariate gaussians
        for (unsigned int a = 0; a < associations.size(); ++a) {
            const double& xLM = map_landmarks.landmark_list[associations[a]-1].x_f;
            const double& yLM = map_landmarks.landmark_list[associations[a]-1].y_f;
            const double& xW = sense_x[a];
            const double& yW = sense_y[a];

            particles[i].weight *= 1.0/(2.0*M_PI*stdx*stdy)*exp(-(((xLM-xW)*(xLM-xW)/(2*stdx*stdx)) + ((yLM-yW)*(yLM-yW)/(2*stdy*stdy))));
            weights[i] = particles[i].weight;
        }
        sum_of_weights += weights[i];
    }

    cout << sum_of_weights << std::endl;

    // normalize weights
    for (unsigned int i = 0; i < particles.size(); ++i) {
        particles[i].weight /= sum_of_weights;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    default_random_engine random_generator;
    std::discrete_distribution<> dist(weights.begin(), weights.end());

    std::vector<Particle> new_particles;
    for(unsigned int i = 0; i < particles.size(); ++i) {
        int drawn_index = dist(random_generator);

        Particle new_particle = particles[drawn_index];
        new_particle.id = i;
        new_particles.push_back(new_particle);
    }

    particles = new_particles;
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
