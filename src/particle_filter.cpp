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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 40	;

	std::random_device rd;
	std::default_random_engine generator(rd());

	weights.resize(num_particles);
	particles.resize(num_particles);

	normal_distribution<double> Guas_x(x, std[0]);
	normal_distribution<double> Guas_y(y, std[1]);
	normal_distribution<double> Guas_theta(theta, std[2]);

	for (int index = 0; index < num_particles; index++)
	{
		particles[index].x = Guas_x(generator);
		particles[index].y = Guas_y(generator);
		particles[index].theta = Guas_theta(generator);
		particles[index].weight = 1.0;
		weights[index] = 1.0;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::random_device rd;
	std::default_random_engine generator(rd());

	normal_distribution<double> Guas_x(0, std_pos[0]);
	normal_distribution<double> Guas_y(0, std_pos[1]);
	normal_distribution<double> Guas_theta(0, std_pos[2]);

	for (int index = 0; index < num_particles; index++)
	{
		if (fabs(yaw_rate) == 0)
		{
			particles[index].x = particles[index].x + velocity* delta_t*cos(particles[index].theta) + Guas_x(generator);
			particles[index].y = particles[index].y + velocity * delta_t*sin(particles[index].theta) + Guas_y(generator);
			particles[index].theta = particles[index].theta + Guas_theta(generator);
		}
		else
		{
			float modifiedAngle = particles[index].theta + yaw_rate * delta_t;
			particles[index].x = particles[index].x + (velocity / yaw_rate)*(sin(modifiedAngle) - sin(particles[index].theta)) + Guas_x(generator);
			particles[index].y = particles[index].y + (velocity / yaw_rate)*(cos(particles[index].theta) - cos(modifiedAngle)) + Guas_y(generator);
			particles[index].theta = particles[index].theta + yaw_rate*delta_t + Guas_theta(generator);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) 
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase
	 
	for (int index = 0; index < predicted.size(); ++index)
	{
		float minDist = INFINITY;
		int indexmin = 0;

		for (int indexI = 0; indexI < observations.size(); ++indexI)
		{
			float DiffX = predicted[index].x - observations[indexI].x;
			float DiffY = predicted[index].y - observations[indexI].y;

			float distance = sqrt(DiffX*DiffX + DiffY* DiffY);

			if (minDist > distance)
			{
				minDist = distance;
				indexmin = indexI;
			}
		}

		observations[indexmin].id = predicted[index].id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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
	for (int indexParticle = 0; indexParticle < num_particles; indexParticle++)
	{
		// we make sure that our weights is reinialized
		particles[indexParticle].weight = 1.0;
		Particle Par = particles[indexParticle];

		//search for the landmark in the range of the perticle
		vector<LandmarkObs> predicted;
		for (int index = 0; index < map_landmarks.landmark_list.size(); index++)
		{
			LandmarkObs landmark = LandmarkObs{ map_landmarks.landmark_list[index].id_i, map_landmarks.landmark_list[index].x_f, map_landmarks.landmark_list[index].y_f };
			double distance = dist(Par.x, Par.y, landmark.x, landmark.y);
			if (distance < sensor_range)
			{
				predicted.push_back(landmark);
			}
		}

		// transform our observations using homogenous transformation
		vector<LandmarkObs> obsserv_modified;
		double cosTheta = cos(Par.theta);
		double sinTheta = sin(Par.theta);

		for (int index = 0; index < observations.size(); index++)
		{
			LandmarkObs Temp;
			Temp.x = observations[index].x * cosTheta - observations[index].y * sinTheta + Par.x;
			Temp.y = observations[index].x * sinTheta + observations[index].y * cosTheta + Par.y;
			obsserv_modified.push_back(Temp);
		}

		// create data assosiation between observations and actual landmarks
		dataAssociation(predicted, obsserv_modified);

		// update the weights using the gaussian probability function 
		double sigma_xx = std_landmark[0]*std_landmark[0];
		double sigma_yy = std_landmark[1] * std_landmark[1];
		double norm = 1.0 / 2.0 * M_PI*std_landmark[0] * std_landmark[1];

		for (int indexI = 0; indexI < obsserv_modified.size(); indexI++)
		{
			int observed_ID = obsserv_modified[indexI].id;

			for (int indexJ = 0; indexJ < predicted.size(); indexJ++)
			{
				int perdicted_ID = predicted[indexJ].id;
				
				if (observed_ID == perdicted_ID)
				{
					double diff_X = obsserv_modified[indexI].x - predicted[indexJ].x;
					double diff_Y = obsserv_modified[indexI].y - predicted[indexJ].y;

					double GausProb = exp(-0.5 * ((diff_X*diff_X)/ sigma_xx) * ((diff_Y*diff_Y)/ sigma_yy)) ;
					Par.weight = Par.weight * GausProb;
				}
			}
		}
		weights[indexParticle] = Par.weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	// setpup the random number generator
	std::random_device rd;
	std::default_random_engine generator(rd());

	// inialized the discrete_distribution
	std::discrete_distribution<> randomWeight(weights.begin(), weights.end());
	
	//create list of new particles
	std::vector<Particle> newparticles;
	newparticles.resize(num_particles);
	
	// randomly choosing the new particles according to current weights
	for (int index = 0; index < num_particles; index++)
	{
		int id = randomWeight(generator);
		newparticles[index] = particles[id];
	}

	//update the particles
	particles = newparticles;

	//clear the old weights
	weights.clear();
	weights.resize(num_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

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
