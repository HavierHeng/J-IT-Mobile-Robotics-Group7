# Design of Part 2 solution - Simple Version using Rtabmap, no funny online path planning

## To run Part 2
1. Put folder into your workspace under `/<ros2_ws>/src/`

2. Colcon build the new package: `colon build --symlink-install --packages-select track_particle_localization`
    - Recommendation: Avoid building all packages, as it may include extra things already built - Zed2 takes quite long to do so.

3. Open a new terminal. Source the workspace overlay if not already added to `.bashrc` : `source ~/ros_ws/install/local_setup.bash`

4. Start remote control of RC Car
    - open terminal: `remote_control`
    - press reset button of arduino zero board
    - Option 1:
        - open new terminal:`ros2 run joy joy_node`
        - open new terminal type: `ros2 run car_control car_control`

    - Option 2:
        - Run this command in terminal: `ros2 launch car_control car_control.launch.py`
  
5. For bounding box visualizer just use the one given in the practical and start rviz. Change the `common.yaml` file in the Zed2 package to use the bag classification model. In new terminals for each command:
    - Start the given visualizer for bounding boxes: `ros2 run obj_det_visualizer obj_visualizer`


    - Start the Zed2 camera: `ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2`
    - Start Rtabmap with the known map after: `ros2 something` - that one ya know with the rviz:=true and rtabmapviz:=true
        - In RViz2, you can add the bounding boxes and shadow (this one is to show the corrected position of the object we use). We also publish path of our bot to go to waypoints, but you can what waypoints its trying to go for via the logs too.

6. Start the node to control the car: `ros2 run track_particle_localization simple`
    - Make sure the car is in autonomous mode by pressing the joycon O 

7. Utilities (required for debugging and setting up waypoint controls):
    - Waypoint crafting: Run `ros2 run pose_logger pose_listener`, and then open Rtabmap and Rviz in localization mode. Use "Get 2D Goal Pose" to draw arrows. Closing the pose_logger will create the yaml in whatever folder you are in.
    - Tuning DBSCAN with matplotlib (+ visualization for whatever objects were observed for debugging): `python3 analyze_dbscan.py path/to/all_observations.csv --show --eps 0.5 --min_samples 2 --min_variance 0.01`

## Description of problem

The problem consists of driving a robot around - either by hand or autonomously (though that is not the case for us). Then as the robot drives around, it will encounter objects of varying classfications - the goal is to return a CSV of most likely object classifications and positions in world map.

To solve these however, we need two things:
1) Where is the robot position in the world frame? This is the reason why we cannot just trust the `ObjectsStamped` position returned by Zed2, its because its Odometry is messed up. Meanwhile, if we use a particle filter to re-estimate our position, then it would go beyond just trusting the virtual odometry, but instead counter checking the odometry against the known map.
2) Where is the Object of classification X in the camera frame? The reason why we turn it back to the camera frame is so we can do a transformation to be relative to the world frame based on the robot position estimate using (1). Internally, zed2 transforms the objects it detects to world frame -> But this uses the shitty estimate of Odometry.

There is also a need for a way for us to control our robot to autonomously get to a position (no online path planning - only offline)
1) Waypointing issues - How to mark our own custom waypoints
    - Optimally, we want our robot to zig zag into custom poses, this way we can see as much of the object
    - If using the `mapGraph` path, do we want to smooth that path (or just not use it in the first place)
2) Moving to waypoint - Tuning the controls to be able to move to said waypoints using localization coordinates from `/rtabmap/localization_pose`

## New Problem breakdowns

### Using Rtabmap for Localization (and why) 
`zed/zed_node/obj_det/objects` is using a shitty estimate of position of object via local visual inertia odometry. It may seem to return in world frame, but this is always estimated via the odom transform with a huge drift value.

1) Convert its objects back to be relative to camera frame - to get a translation/rotation of the object wrt to our camera
2) Then we can calculate the more accurate position of the object in the map frame since we know the position of the robot in map frame from `rtabmap/localization_pose`, we can transform the position of the object from camera frame to map frame (which is a more accurate value).

### Waypoints - how to mark and load them?

Our team uses a custom utility made by ourselves called `pose_logger`.

Because we know that in the RViz launched by Rtabmap, we can "Publish 2D Goal Pose" and then see this `Pose` via `/rtabmap/goal` , we generate a custom YAML file with a few fields (some of them are for behaviour control later):
- id: To name/order our waypoints. Can be changed for ease of use
- action: Behaviour control, changed by hand to mark robot behaviours. "navigate" means that when reaching waypoint, immedately follow to next waypoint. "stop", when reach waypoint, robot pauses.
- point (x,y,z): 3 different fields, for the `Point` x, y, z
- quaternion (w, x, y, z): 4 different fields, for the `Quaternion` w, x, y, z

The last waypoint marked is always assumed to be the final goal pose.

### Waypoints - how to get there

We use a custom utility to grab "Get 2D Pose" from RViz and save them as a yaml file of waypoints.

This part needs us to tune 2 simple P controllers for linear and angular velocity based on the desired position and angle at waypoint
- This is as our car has some drift, we need a P controller to at least pull it back to the waypoint.

The linear.x are capped to 1.0, while angular.z are capped to -3.0 to 3.0. These were fixed in the microros code in the arduino.

Car also has some funny issue with reversing. We have reverse logic if the desired waypoint is behind the car (in the back pi/2 radians)

The movement to the waypoint always uses the RTabmap's better estimate of the robot pose. This helps us stop whereever we want as long as the P controller is tuned.

### Static world assumption - dynamic objects as noise rather than objects
If we assume static world assumption, this means that observed objects with a velocity higher than some threshold is considered an invalid candidate.

We seen this issue with runners on the track. We only want to track non-moving Persons. We will then filter them out in the actual run

Calculate the velocity of objects by tracking their positions over a sliding window:
1) See if there is a minimum number of observations of said object in the sliding window. 
2) Estimate the velocity of objects
- If there are multiple objects of same class, to tell if the objects are the same or different, we can use a small threshold of position between old and new positions

### Data processing and Clustering Tuning

Since we want to know what DBScan cluster size to scan, but it would be stupid to keep running the robot to figure out this value.

Drive around and collect data on observations, Plot on matplotlib the global real world positions of all things seen using all_observations.csv and best_observations.csv. (Debug mode for car will generate a lot of data for visualizations, including velocities of any objects)

After the robot generates two `.csv` files, one for all observations, another for the summarized best observations, we can use the all observation ones to try to get it to observe clustering.
- Plot 1: Fit Global real world positions with different DBSCAN sizes 
- Plot 2 (Prove batched DBSCAN sucks): Scatter plot against unique timestamp of scan they were seen in - so we can see how many objects were seen at each scan and prove that batching the DBSCAN instead of doing a global DBSCAN clustering over the final set is just better


### Covariance and variance to summarize the final values to CSV

Get the mean and covariance of the robot's localisation in the world, using Rtabmap which also returns a `PoseWithCovarianceStamped` from `rtabmap/localization_pose` topic, and pick the most trusted estimate of the detected object position in the world via adding the covariances of the measurement of the objects (`ObjectsStamped`) from `zed/zed_node/obj_det/objects` and the covariance from the position `/rtabmap/localization_pose`.

The math works via transforming all the covariance matrices to the same map frame for the observations (originally in `odom` frame due to Zed2) and camera (originally already in `map` frame due to RTabmap), account for:
- obj_cov: Uncertainty of pos_odom in odom frame.
- robot_cov: Uncertainty of t_mc and R_mc in map frame (from RTabMap).
- odom_cov: Uncertainty of t_oc and R_oc in odom frame (from ZED2’s /zed/zed_node/pose

We can do this by applying transforms to the covariances until all are in map frame. Take into account changing the object in camera frame -> object in odom frame -> object in map frame. Then this can be added to the covariance of the robot position in map.

Then if the variance (`tr(covariance)`) of the resulting covariance of the observation of an object is < the previous variance then update the entry in in the dictionary. The only problem is how to distinguish two objects of the same class detected, but this can be roughly figured out by the clustering the positions of the objects seen before saving into a CSV.

### Temporal filtering

That's right, unlike all other groups we don't just filter by a min/max distance by Zed2 to drop off people.
We in fact also filter by the velocity of objects and drop moving objects, since we can assume all objects in our world are static.
This works via a sliding window of observations.
You can see this in the debug messages as our node runs.

The question is how do you know that an object is the same object if all you have are labels of objects?
- Simply hardcode a threshold for max change to a position from previous timestamp before you consider it a separate entity.

This was tested by having a runner run past us and then tuning the values for max velocity and threshold change of position for an object. 
No false positives, get rekt noise from moving objects.

# Tuning the DBScan to collapse all observations into candidate points
## Usage of our matplotlib utility plotter
To plot the graph against all_observations.csv
`python3 analyze_dbscan.py ~/ros2_ws/all_observations.csv --show --eps 0.5 --min_samples 2 --min_variance 0.01`

This allows you to see what candidate clusters were considered and what is the lowest variance of the cluster.

You can use this script to figure out the values of hyperparams for DBScan (think of it like a cluster radius) and minimum samples

There are two samples in the `csv_output` folder. These were real world recordings of people in the scene
1) Scene 1: Two people (one near, one far)
2) Scene 2: Two people (one near, one far) and 1 Vehicle at the front


# Some observations and decision on DBScan values
## Epsilon / Radius of cluster
Use a small value to prevent clustering everything into the same cluster but not too small such that it makes too many clusters. 

For the race track challenge, 0.5 does so well. In real since the goal point is like x=2.5, by some common sense scaling this kind makes sense between the distance of each object.

## Min samples in cluster
Min samples should be somewhat small, as the car sometimes doesn't see enough of the person. We set to 2 (1 is alright ish, but dangerous)

## Which point represents the cluster - the min variance from our CSV or the closest point to centroid.

Closest point to centroid is 100% better. Min variance is subject to outliers as we saw in one of our tests.

## Min variance 
Some cluster min variance is really high. From our observation, these usually represent outlier e.g smearing due to difficulty for YOLOv8 to see bounding box. As such, even though the cluster has points, we will want to drop the cluster altogether. This min variance threshold that works on our training sets is 0.01.

