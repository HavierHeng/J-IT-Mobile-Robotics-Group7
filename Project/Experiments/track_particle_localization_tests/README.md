# Design of Part 2 solution - Simple Version using Rtabmap, no funny online path planning

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
Because we know that in the RViz launched by Rtabmap, we can "Publish 2D Goal Pose" and then see this `Pose` via `/rtabmap/goal` , we generate a custom YAML file with a few fields (some of them are for behaviour control later:
- id: To name/order our waypoints
- Point (x,y,z): 3 different fields, for the `Point` x, y, z
- Quaternion (w, x, y, z): 4 different fields, for the `Quaternion` w, x, y, z
- action: 

### Waypoints how to get there
This part needs us to tune 2 simple P controllers for linear and angular velocity based on the desired position and angle at waypoint
- This is as our car has some drift


### Static world assumption - dynamic objects as noise
If we assume static world assumption, this means that observed objects with a velocity higher than some threshold is considered an invalid candidate.

We can use a temporal filter to check this based on the timestamp of observation. Then calculate the velocity of objects by a naive check of if class matches up

### Data processing and Clustering

Since

### Covariance and variance to summarize the final values to CSV

Get the mean and covariance of the robot's localisation in the world, using Rtabmap which also returns a `PoseWithCovarianceStamped` from `rtabmap/localization_pose` topic, and pick the most trusted estimate of the detected object position in the world via adding the covariances of the measurement of the objects (`ObjectsStamped`) from `zed/zed_node/obj_det/objects` and the covariance from the position `/rtabmap/localization_pose`.

Then if the variance (`tr(covariance)`) of the resulting covariance of the observation of an object is < the previous variance then update the entry in in the dictionary. The only problem is how to distinguish two objects of the same class detected, but this can be roughly figured out by the clustering the positions of the objects seen before saving into a CSV.


