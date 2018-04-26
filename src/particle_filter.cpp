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
  num_particles=100;
  default_random_engine gen;
  normal_distribution<double> Nx(x,std[0]);
  normal_distribution<double> Ny(y,std[0]);
  normal_distribution<double> Ntheta(theta,std[0]);
  for(int i=0;i<num_particles;i++){
    Particle particle;
    particle.id=i;
    particle.x=Nx(gen);
    particle.y=Ny(gen);
    particle.theta=Ntheta(gen);
    particle.weight=1;
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/ 
  default_random_engine gen;
  for(int i=0;i<num_particles;i++){
    Particle &p=particles[i];
    double new_x;
    double new_y;
    double new_theta;
    if(yaw_rate==0){
      new_x=p.x+velocity*delta_t*cos(p.theta);
      new_y=p.y+velocity*delta_t*sin(p.theta);
      new_theta=p.theta;
    }
    else{
      new_x=p.x+velocity/yaw_rate*( sin(p.theta+yaw_rate*delta_t)-sin(p.theta));
      new_y=p.y+velocity/yaw_rate*(-cos(p.theta+yaw_rate*delta_t)+cos(p.theta));
      new_theta=p.theta+yaw_rate*delta_t;
    }
    normal_distribution<double> Nx(new_x,std_pos[0]);
    normal_distribution<double> Ny(new_y,std_pos[0]);
    normal_distribution<double> Ntheta(new_theta,std_pos[0]);
    particles[i].x=Nx(gen);
    particles[i].y=Ny(gen);
    new_theta=Ntheta(gen);
    //while(new_theta> M_PI) new_theta-=2.0*M_PI;
    //while(new_theta<-M_PI) new_theta+=2.0*M_PI;
    particles[i].theta=new_theta; 
  }
}

inline double distance(LandmarkObs &o,const Map::single_landmark_s &l){
 double dx=o.x-l.x_f;
 double dy=o.y-l.y_f;
 return sqrt(dx*dx+dy*dy);
}

//void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
void ParticleFilter::dataAssociation(const Map &map_landmarks, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for(unsigned int i=0;i<observations.size();i++){
    LandmarkObs &o=observations[i];
    double minDist=1e300;
    int    minIdx =-1; 
    for(unsigned int j=0;j<map_landmarks.landmark_list.size();j++){
      const Map::single_landmark_s &l=map_landmarks.landmark_list[j];
      double dist=distance(o,l);
      if(dist<minDist){
         minDist=dist;
         minIdx=l.id_i;
      }
    }
    //cout<<"minDist="<<minDist<<endl;
    //cout<<"minIdx ="<<minIdx <<endl;
    o.id=minIdx;
  }
}

inline LandmarkObs fromLocalToGlobalTransform(Particle &p,const LandmarkObs &o){
  LandmarkObs to;
  to.x=p.x+o.x*cos(p.theta)-o.y*sin(p.theta);
  to.y=p.y+o.x*sin(p.theta)+o.y*cos(p.theta);
  return to;
}

inline vector<LandmarkObs> observationsFromLocalToGlobalTransform(Particle &p,const vector<LandmarkObs> &observations){
  vector<LandmarkObs> tObservations;
  LandmarkObs to;
  for(unsigned int i=0;i<observations.size();i++){  
    const LandmarkObs &o=observations[i];
    to=fromLocalToGlobalTransform(p,o);
    tObservations.push_back(to);
  }
  return tObservations;
}

long double workOutWeight(LandmarkObs &o,const Map::single_landmark_s &mu,double std_landmark[]){
 double &sx=std_landmark[0];
 double &sy=std_landmark[1];
 double sx2=sx;
 double sy2=sy;
 double dx=o.x-mu.x_f;
 double dy=o.y-mu.y_f;
 double dx2=dx*dx;
 double dy2=dy*dy;
 double e=dx2/(2.0*sx2)+dy2/(2.0*sy2);
 long double p=1.0/(2.0*M_PI*sx*sy)*exp(-e);
 //if(p==0)
  //cout<<"p="<<p<<"dx2="<<dx2<<"dy2="<<dy2<<"sx2"<<sx2<<"sy2"<<sy2<<"e="<<e<<endl;
 return p;
}
Map::single_landmark_s findLandmark(const Map &map_landmarks,int id){
  for(unsigned int i=0;i<map_landmarks.landmark_list.size();i++){
    const Map::single_landmark_s &l=map_landmarks.landmark_list[i];
    if(l.id_i==id)
      return l;
  }
  cout<<"landmark Id not found!!!"<<endl;
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
        //
  // Foreach particle in particles
  //  Foreach observation in observations
  //  1.- Transform to global coordinates the observation from particle
  //  2.- Get association with map_landmarks
  //  3.- Compute weights for particle
  for(int i=0;i<num_particles;i++){
    Particle &p=particles[i];
    //global Observations
    vector<LandmarkObs> gObservations;
    gObservations=observationsFromLocalToGlobalTransform(p,observations);
    dataAssociation(map_landmarks,gObservations);
    p.weight=1;
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
    for(unsigned int j=0;j<gObservations.size();j++){
      LandmarkObs &go=gObservations[j];
      Map::single_landmark_s l=findLandmark(map_landmarks,go.id);
      long double prob=workOutWeight(go,l,std_landmark);
      if(prob>0.0)
       p.weight*=prob;
      else{
       p.weight*=0.00000001;
      }
      p.associations.push_back(go.id);
      p.sense_x.push_back(go.x);
      p.sense_y.push_back(go.y);
    }
    weights[i]=p.weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  discrete_distribution<int> distribution(weights.begin(),weights.end());
  vector<Particle> resample_particles;
  for(int i=0;i<num_particles;i++)
   resample_particles.push_back(particles[distribution(gen)]);
  particles=resample_particles;
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
