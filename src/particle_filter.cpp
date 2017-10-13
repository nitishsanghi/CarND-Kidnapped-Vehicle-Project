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
        //Adding Noise distributions
        default_random_engine gen;
        normal_distribution<double> dist_x(x, std[0]);
        normal_distribution<double> dist_y(y, std[1]);
        normal_distribution<double> dist_t(theta, std[2]);
    
        //Number of particles
        num_particles = 100;
    
        //Initializing particles location and orientation with distributions
        for(int i = 0;i<num_particles;i++){
            weights.push_back(1);
            Particle p;
            p.x = dist_x(gen);
            p.y = dist_y(gen);
            p.theta = dist_t(gen);
            p.weight = 1.0;
            particles.push_back(p);
        }
    
        //Setting initialization to true
        is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    //Gaussian Noise Distributions
    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_t(0, std_pos[2]);
    
    //Prediction for yaw_rate greater than ~0
    if(fabs(yaw_rate)>.001){
        for(int i = 0;i<num_particles;i++){
            double px = particles[i].x;
            double py = particles[i].y;
            double pt = particles[i].theta;
            
            particles[i].x = px + velocity/yaw_rate*(sin(pt + yaw_rate*delta_t)-sin(pt)) + dist_x(gen);
            particles[i].y = py + velocity/yaw_rate*(-cos(pt + yaw_rate*delta_t)+cos(pt)) +dist_y(gen);
            particles[i].theta = pt + yaw_rate*delta_t + dist_t(gen);
            }
    }
    
    //Prediction for yaw_rate = 0
    else{
        for(int i = 0;i<num_particles;i++){
            double px = particles[i].x;
            double py = particles[i].y;
            double pt = particles[i].theta;
            
            particles[i].x = px + velocity*cos(pt)*delta_t  + dist_x(gen);
            particles[i].y = py + velocity*sin(pt)*delta_t + dist_y(gen);
            particles[i].theta = pt  + dist_t(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    /*
     
     Similar to code snippet used below to identify closed neighbor
     
     
     */
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
   
    //For each particle weights will be updates
    //Steps to be taken: Select particle; Transform observation to Map Coordinate; Associate nearest neighbor after transformation using map landmarks; find probability using multi variable gaussian distribution
    for (int i=0; i < num_particles; i++) {
        //Particle selected
        double px = particles[i].x;
        double py = particles[i].y;
        double pt = particles[i].theta;
        long double probw = 1.0;
        
        for (int j=0; j < observations.size(); j++) {
            //Observation selected
            double ox = observations[j].x;
            double oy = observations[j].y;
            //Map coordination transform
            double pmx = ox*cos(pt) - oy*sin(pt) + px;
            double pmy = ox*sin(pt) + oy*cos(pt) + py;
            double landmarkx;
            double landmarky;
            //Nearest neighbor distance placeholder
            double cal_distance = 0.0;
            //Sensor range defines sensor characteristic distance below which nearest neighbor should be found
            double clo_landmark_dis = sensor_range;
           
            for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
                
                //L2 norm/Euclidean distance for nearest neighbor
                cal_distance = sqrt(pow(pmx - map_landmarks.landmark_list[k].x_f,2) + pow(pmy - map_landmarks.landmark_list[k].y_f,2));
                
                //Distance measurement
                if (cal_distance < clo_landmark_dis) {
                    clo_landmark_dis = cal_distance;
                    landmarkx = map_landmarks.landmark_list[k].x_f;
                    landmarky = map_landmarks.landmark_list[k].y_f;
                }
            }
            //Probability function defining weight
            double prob = .5*exp(-0.5*(pow((pmx - landmarkx)/std_landmark[0],2) + pow((pmy - landmarky)/std_landmark[1],2)))/M_PI*(std_landmark[0] * std_landmark[1]);
            
            //Accumulating all the weights
            probw *= prob;
        }
        //Updating weights
        particles[i].weight = probw;
        weights[i] = probw;
    }
}

void ParticleFilter::resample() {
    //This was interesting. Was looking for this function for the quiz. :)
    vector<Particle> resparticles;
    default_random_engine gen;
    
    discrete_distribution<int> dd(weights.begin(), weights.end());
    
    for (int i=0; i<num_particles;++i){
        resparticles.push_back(particles[dd(gen)]);
    }
    
    particles = resparticles;

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
