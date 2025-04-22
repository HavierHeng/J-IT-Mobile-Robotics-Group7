#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
import tf_transformations as tr
from std_msgs.msg import String, Header, ColorRGBA
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from math import sqrt, cos, sin, pi, atan2
from threading import Thread, Lock
from math import pi, log, exp
from builtin_interfaces.msg import Time, Duration
import random
import numpy as np
import sys
import pickle


class Particle(object):
    def __init__(self, id, x,y, theta):
        self.x = x
        self.y = y
        self.id = id
        self.theta = theta

    def copy(self):
        return Particle(self.id, self.x, self.y, self.theta)
    
class ParticleFilter(object):
    def __init__(self, num_particles, occ_grid_map, xmin, xmax, ymin, ymax,
                 laser_min_range, laser_max_range, laser_min_angle, laser_max_angle,
                 dynamics_translation_noise_std_dev,
                 dynamics_orientation_noise_std_dev,
                 beam_range_measurement_noise_std_dev, node, n_thresh=0):
        
        self.num_particles = num_particles
        self.n_thresh = n_thresh
        self.ogm = occ_grid_map
        self.grid_map = np.array(self.ogm['data'], dtype='int8')
       

        height = self.ogm['info'].get('height', 0)
        width = self.ogm['info'].get('width', 0)
        self.grid_map = self.grid_map.reshape((height, width)) 
        self.height = height
        self.width = width
        
        self.grid_bin = (self.grid_map == 0).astype('uint8')  # Cell is True iff probability of being occupied is zero

        # Workspace boundaries - in terms of metric (not the grid coords)
        self.xmax = xmax
        self.xmin = xmin
        self.ymin = ymin
        self.ymax = ymax
        
        self.laser_max_angle = laser_max_angle
        self.laser_min_angle = laser_min_angle
        self.laser_max_range = laser_max_range
        self.laser_min_range = laser_min_range

        # Std deviation of noise affecting translation in the dynamics model for particles 
        self.dynamics_translation_noise_std_dev = dynamics_translation_noise_std_dev

        # Std deviation of noise affecting orientation in the dynamics model for particles 
        self.dynamics_orientation_noise_std_dev = dynamics_orientation_noise_std_dev

        # Std deviation of noise affecting measured range from the laser measurement model  
        self.beam_range_measurement_noise_std_dev = beam_range_measurement_noise_std_dev

        # Number of laser beams to simulate when predicting what a
        # particle's measurement is going to be
        self.eval_beams = 9

        # Previous odometry measurement of the robot
        self.last_robot_odom = None

        # Current odometry measurement of the robot 
        self.robot_odom = None

        # Relative motion since the last time particles were updated
        self.dx = 0
        self.dy = 0
        self.dyaw = 0

        self.particles = []
        self.weights = []

        # ROS Node
        self.node = node

    def get_random_free_state(self):
        """
        Helper function to get a random free state in the map (i.e occupany grid map cell not taken) for particle initialization reasons.
        """
        while True:
            # Note: we initialize particles closer to the robot's initial
            # position in order to make the initialization easier
            xrand = np.random.uniform(self.xmin, self.xmax)
            yrand = np.random.uniform(self.ymin, self.ymax)
            row, col = self.metric_to_grid_coords(xrand, yrand)
            if self.grid_bin[row, col]:
                # TODO: Bonus Part B - Randomizing the particles' orientation in any arbitrary direction
                # theta = np.random.uniform(0, 0.01)
                theta = np.random.uniform(0, 2*pi)
                return xrand, yrand, theta
        
    def init_particles(self):
        """Initializes particles uniformly randomly with map frame coordinates, 
        within the boundaries set by xmin,xmax, ymin,ymax"""
        for i in range(self.num_particles):
            xrand, yrand, theta = self.get_random_free_state()
            # Note: same orientation as the initial orientation of the robot
            # to make initialization easier
            self.particles.append(Particle(i, xrand, yrand, theta))

    @staticmethod
    def exponentiate_and_normalize(x):
        b = x.max()
        x = np.exp(x - b)
        return x / x.sum()

    def handle_observation(self, laser_scan, dt):
        """
        Does prediction, weight update, and resampling on new laser_scan observation. 

        Also takes into account changes in dynamics of robot since last laser_scan observation as that is needed to update all particles dynamics by the same amount as the robot.

        Weights are normally distributed with a sum of 1. This is done by converting all prediction errors (between bot and particle beams) into points on a Gaussian dist (via the normal dist formula) and normalizing by the sum of all the points to get their total sum to 1.

        The check of Effective Sample Size (ESS) just means how diverse the particle set is, since it means how many independent samples would be needed to achieve the same level of accuracy as the weighted samples. We don't want to resample if it drops too low as that indicates particle degeneracy - a few particles are dominating with high weights, the rest have weights near zero.

        n_thresh=0 disables the ESS check (default behaviour)
        """
        # Task: For every particle:
        # 1) Predict its relative motion since the last time an observation was received using predict_particle_odometry().
        # 2) Compute the squared norm of the difference between the particle's predicted laser scan and the actual laser scan 
        errors = []

        for particle in self.particles:
            # Predict particle motion based on odom - basically update each particle with dx, dy, the dynamics changed from last reading
            self.predict_particle_odometry(particle, dt)

            # Get the squared norm error between particle's predicted scan and actual scan - needed to filter out the bad particle candidates with laser scans that don't match up that well with the actual robot laser scan measurements
            error = self.get_prediction_error_squared(laser_scan, particle)
            errors.append(error)

        errors = np.array(errors)

        # Task : exponentiate the prediction errors you computed above
        # using numerical stability tricks such as
        # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick
        # This is already given by exponentiate_and_normalize()
        # Smaller errors == higher weights to be resampled to, so negative
        # Also weights sum to probability of 1: property is needed for us to pull off stochastic sampling later in resample()
        neg_errors = -errors / (2 * self.beam_range_measurement_noise_std_dev ** 2)  # Gaussian formula exponential part
        self.weights = self.exponentiate_and_normalize(neg_errors)

        # Effective sample size: https://stonesoup.readthedocs.io/en/latest/auto_tutorials/04_ParticleFilter.html#use-of-effective-sample-size-resampler-ess
        # Used as a check to check if it is needed to resample particles since low ESS values means the particle set is not diverse, and PF is about to degenerate
        N_eff = 1.0 / np.sum(np.square(self.weights))
        print("effective sample size", N_eff)

        # Task: Do resampling. Depending on how you implement it you might or might not need to normalize your weights by their sum, so they are treated as probabilities
        # Resample only if eff sample size < threshold, to prevent particle degeneracy issues when most particles end up too concentrated
        # n_thresh of 0 disables the ESS check
        if N_eff < self.n_thresh or self.n_thresh == 0:
            self.resample()

    def resample(self):
        """
        Implements resampling in particle filters. 

        Task: sample particle i with probability that is proportional to its weight w_i. 

        Sampling can be done with repetition/replacement, so you can sample the same particle more than once.

        Note: This is the 2nd stochastic universal sampling (SUS) algorithm from the notes. 
        In essence, we build a Probability mass function (PMF) which represents the pie chart thingy, pick a random offset for the first pointer, and all subsequent pointers are equally spaced until it wraps around the pie chart.
        """

        n_particles = len(self.particles)
        new_particles = []  # After resampling based on probs/weight

        # Sum of all weights - PMF function
        # basically the pie chart thing
        cumulative_sum = np.cumsum(self.weights)

        # Small initial random offset - think of it as the position of the first pointer in the pie chart
        # Following pointer positions will be: (offset + i/N) until it loops around the pie chart
        r = random.random() / n_particles

        # SUS until all particles have been resampled
        for i in range(n_particles):
            u = r + i / n_particles  # u represents the current pointer weight
            
            # Find particle index corresponding to pointer weight
            idx = 0
            while idx < n_particles - 1 and u > cumulative_sum[idx]:
                # Keep searching until the first particle bin that is over the checking pointer weight
                idx += 1
            # Copy found particle index
            new_particles.append(self.particles[idx].copy())

        self.particles = new_particles

    def simulate_laser_scan_for_particle(self, x, y, yaw_in_map, angles, min_range, max_range):
        """
        If the robot was at the given particle, what would its laser scan
        be (in the known map)? 

        Returns the predicted laser ranges if a particle with state (x,y,yaw_in_map) is to scan along relative angles in angles.

        Task: for every relative angle in angles
        1. Compute the absolute angle based on the robot's orientation
        2. Do ray tracing from (x,y) along the absolute angle using step size range_step 
           (a) If the currently examined point is within the bounds of the workspace: stop if it meets an obstacle or if it reaches max_range
           (b) If the currently examined point is outside the bounds of the workspace: stop if it reaches max_range
        3. Return the computed collection of ranges corresponding to the given angles
        """
        
        ranges = []
        range_step = self.ogm['info']['resolution']  # meters per cell, affects how far each ray jumps per ray casting check

        # Ray trace per angle
        for angle in angles:
            # Absolute angle wrt to the particle frame
            abs_angle = yaw_in_map + angle

            r = min_range

            # Ray trace until max_range reached
            while r < max_range:
                # Where is point at current ray range (in metric)
                px = x + r * cos(abs_angle)
                py = y + r * sin(abs_angle)

                # Condition (b): out of bounds of workspace (self.xmin, xmax, ymin, ymax)
                # Note: xmin, xmax, ymin and ymax are in metric coords
                if (px < self.xmin or px > self.xmax or py < self.ymin or py > self.ymax):
                    r = max_range
                    break

                # Condition (a): within bounds, then check if obstacle has been met or if ray reaches max_range
                row, col = self.metric_to_grid_coords(px, py)

                # Is the cell occupied? i.e obstacle collision
                # Use grid_map over grid_bin -> since anything that is maybe occupied i.e in range (0, 100) is not for certain an obstacle yet
                if self.grid_map[row, col] == 100:
                    break

                r += range_step
            ranges.append(min(r, max_range))  # Add ray

        return ranges

    def subsample_laser_scan(self, laser_scan_msg):
        """Subsamples a set number of beams (self.eval_beams) from the incoming actual laser scan. It also
        converts the Inf range measurements into max_range range measurements, in order to be able to 
        compute a difference."""
    
        # Just like in the occupancy grid mapping assignment you might need this snippet
        # to convert the laser points from the husky_1/base_laser frame, whose z-axis points downwards
        # to the same frame pointing upwards

        N = len(laser_scan_msg.ranges)
        ranges_in_upwards_baselaser_frame = laser_scan_msg.ranges
        angles_in_baselaser_frame = [(laser_scan_msg.angle_max - laser_scan_msg.angle_min)*float(i)/N + laser_scan_msg.angle_min for i in range(N)]

        step = N/self.eval_beams
        angles_in_upwards_baselaser_frame = angles_in_baselaser_frame[::int(step)]
        ranges_in_upwards_baselaser_frame = ranges_in_upwards_baselaser_frame[::int(step)]

        assert (len(ranges_in_upwards_baselaser_frame) == len(angles_in_upwards_baselaser_frame))
        
        actual_ranges = []
        for r in ranges_in_upwards_baselaser_frame:
            
            if r >= self.laser_min_range and r <= self.laser_max_range:
                actual_ranges.append(r)
            
            if r < self.laser_min_range:
                actual_ranges.append(self.laser_min_range)
            
            if r > self.laser_max_range:
                actual_ranges.append(self.laser_max_range)
        

        return actual_ranges, angles_in_upwards_baselaser_frame
    
    def get_prediction_error_squared(self, laser_scan_msg, particle):
        """
        This function evaluates the squared norm of the difference/error between the  
        scan in laser_scan_msg and the one that was predicted by the given particle. 
        
        Assume that the bearing of each beam relative to the robot's orientation has zero noise, 
        so the only noise in the measurement comes from the range of each beam and is 
        distributed as N(0, beam_range_measurement_std_dev^2)
        """

        # If the particle is out of the bounds of the workspace
        # give it a large error
        if particle.x < self.xmin or particle.x > self.xmax:
            return 300

        if particle.y < self.ymin or particle.y > self.ymax:
            return 300


        # If the particle falls inside an obstacle
        # give it a large error
        row, col = self.metric_to_grid_coords(particle.x, particle.y)
        if row < self.height and col < self.width: 
            if self.grid_map[row, col] == 100:
                return 300
        else:
            return 300
        
        
        assert (self.laser_min_range >= 0)
        assert (self.laser_max_range > 0)

        # Task: subsample the recived actual laser scan using the subsample_laser_scan method above
        # Fixes the min and max range of scans, and takes a smaller subset of laser scan beams for consideration (its based on self.eval_beams)
        # This is since we need to simulate the laser checks from each particle, and it would be ridiculous to consider all N beams from robot
        actual_ranges, angles = self.subsample_laser_scan(laser_scan_msg)

        # Task: simulate a laser scan from particle using one of the methods of this class
        predicted_ranges = self.simulate_laser_scan_for_particle(
                particle.x,
                particle.y,
                particle.theta,
                angles,
                self.laser_min_range,
                self.laser_max_angle)
        
        # Task: compute the difference between predicted ranges and actual ranges. Take the squared norm of that difference
        squared_errors = [(p_range - a_range)**2 for p_range, a_range in zip(predicted_ranges, actual_ranges)]
        error = sum(squared_errors)  # Sum L2 errors of all subsampled simultaed beams from particle (as compared to actual robot)

        # Error is used to calculate how far the particle simulated laser scan is from what is seen by the robot actual laser scan
        return error

    def handle_odometry(self, robot_odom):
        """Compute the relative motion of the robot from the previous odometry measurement
        to the current odometry measurement."""
        self.last_robot_odom = self.robot_odom
        self.robot_odom = robot_odom

        if self.last_robot_odom:
            
            p_map_currbaselink = np.array([self.robot_odom.pose.pose.position.x,
                                           self.robot_odom.pose.pose.position.y,
                                           self.robot_odom.pose.pose.position.z])

            p_map_lastbaselink = np.array([self.last_robot_odom.pose.pose.position.x,
                                           self.last_robot_odom.pose.pose.position.y,
                                           self.last_robot_odom.pose.pose.position.z])

            q_map_lastbaselink = np.array([self.last_robot_odom.pose.pose.orientation.x,
                                           self.last_robot_odom.pose.pose.orientation.y,
                                           self.last_robot_odom.pose.pose.orientation.z,
                                           self.last_robot_odom.pose.pose.orientation.w])

            q_map_currbaselink = np.array([self.robot_odom.pose.pose.orientation.x,
                                           self.robot_odom.pose.pose.orientation.y,
                                           self.robot_odom.pose.pose.orientation.z,
                                           self.robot_odom.pose.pose.orientation.w])
            
            R_map_lastbaselink = tr.quaternion_matrix(q_map_lastbaselink)[0:3,0:3]
            
            p_lastbaselink_currbaselink = R_map_lastbaselink.transpose().dot(p_map_currbaselink - p_map_lastbaselink)
            q_lastbaselink_currbaselink = tr.quaternion_multiply(tr.quaternion_inverse(q_map_lastbaselink), q_map_currbaselink)
            
            _, _, yaw_diff = tr.euler_from_quaternion(q_lastbaselink_currbaselink) 

            dt_since_last_odom = (self.robot_odom.header.stamp.sec - self.last_robot_odom.header.stamp.sec)
            # Calculate time difference in nanoseconds
            dt_since_last_scan_nsec = (self.robot_odom.header.stamp.nanosec - self.last_robot_odom.header.stamp.nanosec)
            # Convert nanoseconds to seconds and add to the time difference in seconds
            dt_since_last_odom += dt_since_last_scan_nsec / 1e9

            self.dyaw = yaw_diff/dt_since_last_odom
            self.dx = p_lastbaselink_currbaselink[0]/dt_since_last_odom
            self.dy = p_lastbaselink_currbaselink[1]/dt_since_last_odom

            
    def predict_particle_odometry(self, particle,dt):
        """
        Where will the particle go after time dt passes?
        This function modifies the particle's state by simulating the effects
        of the given control forward in time. 

        Assume Dubins dynamics with variable forward velocity for the Husky.
        """
 
        nx = random.gauss(0, self.dynamics_translation_noise_std_dev)
        ny = random.gauss(0, self.dynamics_translation_noise_std_dev)
        ntheta = random.gauss(0, self.dynamics_orientation_noise_std_dev)

        v = sqrt(self.dx**2 + self.dy**2)

        # Don't let the particle propagation be dominated by noise
        if abs(v) < 1e-10 and abs(self.dyaw) < 1e-5:
            return
       
        particle.x += v * cos(particle.theta) * dt + nx
        particle.y += v * sin(particle.theta) * dt + ny
        particle.theta += self.dyaw * dt + ntheta

        
    def metric_to_grid_coords(self, x, y): 
        """Converts metric coordinates to occupancy grid coordinates"""
        
        origin_x = self.ogm['info']['origin']['position']['x']
        origin_y = self.ogm['info']['origin']['position']['y']
        resolution = self.ogm['info']['resolution']
        height = self.ogm['info']['height']
        width = self.ogm['info']['width']

        gx = (x - origin_x) / resolution
        gy = (y - origin_y) / resolution
        row = min(max(int(gy), 0), height-1)
        col = min(max(int(gx), 0), width-1)
        return (row, col)
        

class MonteCarloLocalization(Node):
    
    def __init__(self, num_particles, xmin, xmax, ymin, ymax):
      
        super().__init__('monte_carlo_localization')
        #############param decleartion##########################################################
        self.declare_parameter('map_file')
        self.declare_parameter('dynamics_translation_noise_std_dev')
        self.declare_parameter('dynamics_orientation_noise_std_dev')
        self.declare_parameter('beam_range_measurement_noise_std_dev')
        ###########################################################################################
        
        self.map_file = self.get_parameter('map_file').value
        self.dynamics_translation_noise_std_dev = self.get_parameter('dynamics_translation_noise_std_dev').value
        self.dynamics_orientation_noise_std_dev = self.get_parameter('dynamics_orientation_noise_std_dev').value
        self.beam_range_measurement_noise_std_dev = self.get_parameter('beam_range_measurement_noise_std_dev').value
        
        ###################################################################################
        pkl_file = open(self.map_file, 'rb')
        self.ogm = pickle.load(pkl_file)
        pkl_file.close()
        ########################################################################################


        self.q_baselink_baselaser = np.array([1.0, 0, 0, 0])
        self.R_baselink_baselaser = tr.quaternion_matrix(self.q_baselink_baselaser)[0:3,0:3]
        self.p_baselink_baselaser = np.array([0.337, 0.0, 0.308])

        self.pf = ParticleFilter(num_particles, self.ogm, xmin, xmax, ymin, ymax, 0, 0, 0, 0,
                                 self.dynamics_translation_noise_std_dev,
                                 self.dynamics_orientation_noise_std_dev,
                                 self.beam_range_measurement_noise_std_dev, self)
        
        self.pf.init_particles()
        self.last_scan = None
        self.mutex = Lock()
        ################################Publisher##################################################
        self.laser_points_marker_pub = self.create_publisher(Marker,'debug/laser_points',1) 
        self.particles_pub = self.create_publisher(MarkerArray,'particle_filter/particles', 1)

        ########################Subcriber#####################################################
        self.odom_sub = self.create_subscription(Odometry,'odom', self.odometry_callback, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)) #need to change according yor odometey topic
        self.laser_sub = self.create_subscription(LaserScan,'scan', self.laser_scan_callback,QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))  #need to chage according your scan topic
        ######################################newly_added########################################################
       
        self.timer = self.create_timer(1.0, self.publish_particle_markers)

   
     ######################################################################################################################
    def odometry_callback(self, msg):
        self.mutex.acquire()
        self.pf.handle_odometry(msg)
        self.mutex.release()
    
    def get_2d_laser_points_marker(self, timestamp, frame_id, pts_in_map, marker_id, rgba):
        msg = Marker()
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id
        msg.ns = 'laser_points'
        msg.id = marker_id
        msg.type = 6
        msg.action = 0
        msg.points = [Point(x=pt[0],y=pt[1],z=pt[2]) for pt in pts_in_map]
        msg.colors = [rgba for _ in pts_in_map]
        
        for pt in pts_in_map:
            assert((not np.isnan(pt).any()) and np.isfinite(pt).all())
           
        msg.scale.x = 0.1 
        msg.scale.y = 0.1
        msg.scale.z = 0.1

        
        return msg
        
        
    def laser_scan_callback(self, msg):
        self.pf.laser_min_angle = msg.angle_min
        self.pf.laser_max_angle = msg.angle_max
        self.pf.laser_min_range = msg.range_min
        self.pf.laser_max_range = msg.range_max
        #check with ranges
        dt_since_last_scan = 0
        
        if self.last_scan:
            dt_since_last_scan_sec = (msg.header.stamp.sec - self.last_scan.header.stamp.sec)
            # Calculate time difference in nanoseconds
            dt_since_last_scan_nsec = (msg.header.stamp.nanosec - self.last_scan.header.stamp.nanosec)
            # Convert nanoseconds to seconds and add to the time difference in seconds
            dt_since_last_scan_sec += dt_since_last_scan_nsec / 1e9
            dt_since_last_scan = dt_since_last_scan_sec

        self.mutex.acquire()
        self.publish_laser_pts(msg)
        self.pf.handle_observation(msg, dt_since_last_scan)

        self.pf.dx = 0
        self.pf.dy = 0
        self.pf.dyaw = 0
                
        self.mutex.release()
        self.last_scan = msg
      

    def publish_laser_pts(self, msg):
        """Publishes the currently received laser scan points from the robot, after we subsampled
        them in order to comparse them with the expected laser scan from each particle."""
        if self.pf.robot_odom is None:
            return

        subsampled_ranges, subsampled_angles = self.pf.subsample_laser_scan(msg)
        
        N = len(subsampled_ranges)
        x = self.pf.robot_odom.pose.pose.position.x
        y = self.pf.robot_odom.pose.pose.position.y
        _, _ , yaw_in_map = tr.euler_from_quaternion(np.array([self.pf.robot_odom.pose.pose.orientation.x,
                                                               self.pf.robot_odom.pose.pose.orientation.y,
                                                               self.pf.robot_odom.pose.pose.orientation.z,
                                                               self.pf.robot_odom.pose.pose.orientation.w]))
        
        pts_in_map = [ (x + r*cos(theta + yaw_in_map),
                        y + r*sin(theta + yaw_in_map),
                        0.3) for r,theta in zip(subsampled_ranges, subsampled_angles)]

        lpmarker = self.get_2d_laser_points_marker(msg.header.stamp, 'odom', pts_in_map, 30000, ColorRGBA(r=0.0,g=0.0,b=1.0,a=1.0))
      
        self.laser_points_marker_pub.publish(lpmarker)
        
        
    def get_particle_marker(self, particle, marker_id, avg=False):
        """Returns an rviz marker that visualizes a single particle"""
        particle_marker = Marker()
        particle_marker.header.stamp = self.get_clock().now().to_msg() #timestamp
        particle_marker.header.frame_id = 'odom'
        particle_marker.ns = 'particles'
        particle_marker.id = marker_id
        particle_marker.type = Marker.ARROW 
        particle_marker.action = Marker.ADD 
        lifetime_duration = Duration()
        lifetime_duration.sec = 1 
        particle_marker.lifetime = lifetime_duration
        
        yaw_in_map = particle.theta
        vx = cos(yaw_in_map)
        vy = sin(yaw_in_map)

        if avg:
            particle_marker.color = ColorRGBA(r=0.0,g=0.0,b=1.0,a=1.0)
            particle_marker.points.append(Point(x=particle.x, y=particle.y, z=0.3))
            particle_marker.points.append(Point(x=particle.x + 0.3*vx, y=particle.y + 0.3*vy, z=0.3))
        else:
            particle_marker.color = ColorRGBA(r=0.0,g=1.0,b=0.0,a=0.1)
            particle_marker.points.append(Point(x=particle.x, y=particle.y, z=0.2))
            particle_marker.points.append(Point(x=particle.x + 0.3*vx, y=particle.y + 0.3*vy, z=0.2))
        
        particle_marker.scale.x = 0.05
        particle_marker.scale.y = 0.15
        particle_marker.scale.z = 0.1
        return particle_marker

    def publish_particle_markers(self):
        """ Publishes the particles of the particle filter in rviz"""
        
        marker_array = MarkerArray()
        

        
        for i, particle in enumerate(self.pf.particles):
            particle_marker = self.get_particle_marker(particle, i)
            
            marker_array.markers.append(particle_marker)

        pos = np.array([[p.x, p.y, p.theta] for p in self.pf.particles])
        avg_pos = np.mean(pos, axis=0)
        avg_particle = Particle(-1, avg_pos[0], avg_pos[1], avg_pos[2])
        avg_particle_marker = self.get_particle_marker(avg_particle, i+1, avg=True)
        marker_array.markers.append(avg_particle_marker)
           
        self.particles_pub.publish(marker_array)
        
   


def main(args=None):
    
    xmin = -3
    xmax = 3
    ymin = -3
    ymax = 3
    # TODO: Bonus Part B: Set num particles to 2000 particles
    # num_particles = 300
    num_particles = 2000

    rclpy.init(args=args)
    node = MonteCarloLocalization(num_particles, xmin, xmax, ymin, ymax)    
   
    rclpy.spin(node)
    node.node.destroy_node()
    rclpy.shutdown()
            
if __name__ == '__main__':
    main()


