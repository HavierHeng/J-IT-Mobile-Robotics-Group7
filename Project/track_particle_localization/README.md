# Design of Part 2 solution (Good for report too)

## Description

The problem consists of driving a robot around - either by hand or autonomously (though that is not the case for us). Then as the robot drives around, it will encounter objects of varying classfications - the goal is to return a CSV of most likely object classifications and positions in world map.

To solve these however, we need two things:
1) Where is the robot position in the world frame? This is the reason why we cannot just trust the `ObjectsStamped` position returned by Zed2, its because its Odometry is messed up. Meanwhile, if we use a particle filter to re-estimate our position, then it would go beyond just trusting the virtual odometry, but instead counter checking the odometry against the known map.
2) Where is the Object of classification X in the camera frame? The reason why we turn it back to the camera frame is so we can do a transformation to be relative to the world frame based on the robot position estimate using (1). Internally, zed2 transforms the objects it detects to world frame -> But this uses the shitty estimate of Odometry.

## Implementations
Initially our team thought this problem was about monte carlo localization - but this was pretty complex, the logic is still explained below.

But then we realized we could just use Rtabmap for localization (replacing the need for even writing MCL), then track the confidence of an object position via its covariance from the robot's pose and the object measurement. This version also comes with autonomous movement. 
- Either it uses a pre-planned path via `/rtabmap/mapGraph` 
- Or custom waypoints/goals which we set ourselves 
- If we do Rtabmap, we can set points via Publish 2D Points and follow them by subscribing. The challenge is to find out which topic gives us waypoints to track when RTabmap runs in localization mode.
- Some hints: http://official-rtab-map-forum.206.s1.nabble.com/How-to-label-nodes-and-send-goals-from-rtabmap-interface-td903.html
- Rtabmap provides goal points via `/rtabmap/goal_node`.
- For `/rtabmap/goal` topic, this is used for the human controller from RVIZ to tell the robot a goal point, the pose is transformed in map frame before planning, but its not very useful to use given autonomy.


For the SLAM version we explored, we know we have no maps, so it works via building a huge covariance matrix of landmarks as we explore the world. Optimally, the points are double confirmed via loop closure.


## Are you being redundant to do MCL when Zed2 already gives you a position using Visual Inertia Odometry?

VIO (like the ZED2's output) gives you relative pose estimates, usually with high short-term precision but long-term drift due to:
- Sensor noise
- Lack of loop closure (Zed2 odom topic does have it but its not very consistent)
- No absolute reference to the map

This is good enough for local motion (e.g finding out how far you moved from odom/start frame). Accumulates error however.

Hence the position given by Zed2 comes with some uncertainty (`PosewithCovariance`). I can turn these uncertainty into a set of particles to test hypothesis on positions of where the bot can be in the global context based on what they see, therefore fixing up any uncertainties from local odometry.

Other problems these tackle:
1) Global localization: If your robot gets "lost", MCL can recover from bad initial guesses.
2) Map correction: By comparing sensor data (e.g., depth or pointcloud from ZED2) with a known map, MCL can correct for drift.
3) Probabilistic tracking: It maintains multiple hypotheses of position and can weigh them against real-world observations.

MCL/particle filter gives global accuracy of position in world, as compared to VIO/odometry local consistency.

This is painfully slow and has high overhead however so its only computed like 1Hz to confirm the position of the car (Odom is computed via EKF constantly).

## Follow-up after some work on a Particle Filter: But are you being redundant to do MCL when Rtabmap is on

Yup, and our misunderstanding of the capabilities of Rtabmap made us overthink the implementation at first. The idea is the same though, instead of using Particle Filter to manually calculate the mean and covariance of the robot's localisation in the world, we instead use Rtabmap which also returns a `PoseWithCovarianceStamped` from `rtabmap/localization_pose` topic, and pick the most trusted estimate of the detected object position in the world via adding the covariances of the measurement of the objects (`ObjectsStamped`) from `zed/zed_node/obj_det/objects` and the covariance from the position `/rtabmap/localization_pose`.

Then if the variance (`tr(covariance)`) of the resulting covariance of the observation of an object is < the previous variance then update the entry in in the dictionary. The only problem is how to distinguish two objects of the same class detected, but this can be roughly figured out by the clustering the positions of the objects seen before saving into a CSV.

The other part of the problem is that since `zed/zed_node/obj_det/objects` is using a shitty estimate of position in a local odometry (it returns in world frame but its still made via a transforms including the bad odometry from visual interial odometry), we want to convert its objects back to be relative to camera frame. Then we can calculate the more accurate position of the object in the map frame since we know the position of the robot in map frame from `rtabmap/localization_pose`, we can transform the position of the object from camera frame to map frame (which is a more accurate value).

## Sub-Problems (for the very complex initial idea for Particle filtering - before we realized how redundant it was given Rtabmap exists)
### Sub-Problem 1: Localization of robot in a known map
The first sub problem is localization in a known map, by finding the robot's position in the world space based on a known map.

This is solvable via Adaptive Monte Carlo Localization, aka the Particle Filter.

Given the dynamics model, observation model, as the robot moves around, we apply the same dynamics to the particles, taking into account the noise of the dynamics (in `PosewithCovariance`). Then using the observation models, we determine which particle hypothesis are wrong and hence should be resampled. Repeat until it converges onto where the robot roughly is in the world map.

Resample via Stochastic Universal Sampling (SUS AMONGUS ඞ) - this is via the weights we calculated based on how likely to pick these particles with replacement.

#### Sub Sub Problem: Prior or no prior information - to initialize all particles in the direction of the car? Or random orientations?
Since in our race track challenge, there is some prior information that our possible car positions are going to point straight from the start line, it might be faster to just initialize all particles facing straight out, instead of RNG direction. 

It could speed up the point convergence given this information.

#### Sub Sub Problem: How do we know which points in pointcloud map does a particle see?
While we can transform using the Translation matrix and Quaternion to figure where a particle should be looking, how much can the particle see if it had the intrinsic camera parameters of the robot?

The robot might have a certain focal length, principal axis and so on defined by its camera parameters. These have to applied onto the particle as well.

This wasn't an issue in the HW using LaserScan since LaserScan has a fixed number of beams in fixed directions. However, in our visual based system, a frame recorded by the robot can have as many feature depth points as it can pick up using its stereo cameras.

#### Sub Sub Problem: Matching an unequal amount of points seen by a stimulated particle and the real robot
The difference between this and the homework that might pose some issue is "How do we match points seen by the robot with points in known Rtabmap?"

This was not an issue in HW using LaserScan since LaserScan means that there will always be the same number of beams returned by a particle and the actual robot. There is a 1-to-1 correspondence of beams from both sides.

On the other hand, our camera based robot system means that if a particle goes into a position that was out of view for the camera during the mapping phase, it would see less points than the robot. It may also just happen to go into a position that can also see more points on the Rtabmap pointcloud than the robot's pointcloud.

This is a problem - if we want to match points seen by particles in a known pointcloud map against the robot's pointcloud, we normally calculate a least squares of 1-to-1 points from the particle and robot, and use that to scale to Gaussian likelihood for weights for stochastic universal sampling. We also cannot use something like Iterative Closest Point given that we don't know the exact pose of a robot (its a guess based on a lot of particles). 

This is solvable via a means of K-nearest neighbours, which in practice is solved via `KDTree` (`SciPy` has an easy to use implementation). Initially, read in the PointCloud from the RTabMap known map, then save it as a KDTree(points). 

KDTree allows for a fast way that given a point observed by the robot, we can find the closest point in the known map that matches it. Then we can take this point seen by the robot (now in the known map) which has a 1-to-1 correspondence with the points seen by a particle in the known map.

#### Sub Sub Problem: Accounting for degeneracy / wrong hypothesis due to particles converging
This can be solved by reintroducing randomness to particles when resampling.

We can also keep track of degeneracy via effective sample size (ESS):
$$ \text{ESS} = \frac{1}{\sum_{i=1}^N w_i^2} $$
where:
- $w_i$  are the normalized particle weights (i.e., $sum w_i = 1$)
- $N$  is the total number of particles


- $\text{ESS} \approx N$: good distribution, particles are diverse.
- $\text{ESS} \ll N$: degeneracy! Most weights are near-zero.
- A common threshold is $ \text{ESS} < N/2 $, at which point triggers resampling.

```python
def compute_ess(weights):
    return 1.0 / np.sum(np.square(weights))

if compute_ess(weights) < N / 2:
    particles = resample(particles, weights)
    weights = np.ones(N) / N
```

#### Sub Sub Problem: Avoiding false positives when humans walking around
If the robot does see a human walking about, it might be able to track and classify the human as a potential object candidate. 

To prevent this, one method our team has thought about to use is a threshold min and max distance, if an object is detected outside of this threshold distance (distance is an L2 distance based on the `ObjectsStamped`), then it is dropped without being considered. 

This is as objects that are too close could just be due to some temporary images, while objects that are too far are too uncertain in position to be a good consideration for localization.

We also abuse some region of interest, we can mask out background areas using opencv (same as what was done in track following).

#### Sub Sub Problem: Visual Inertial Odometry woes
We know how bad Visual Inertial Odometry based on the Zed2 camera can be. But this isn't a problem. This is as our particle filter does consider the uncertainty of a position of the car to apply the necessary translations and rotations to move each particle. Eventually, any new observations will cause a resampling, killing the bad hypotheis.

#### Sub Sub Problem: /cmd_vel throttle might not be equal to how many m/s in real world, visual odometry as ground truth

While using /cmd_vel works in simulations to update the dynamics model's linear x velocity, this is not the case in a real robot. Putting our throttle to the max value of 1.0 might not correspond 1:1 to the 1.0m/s in physical world assumption. 

Also because when we are testing the robot without battery, our robot is not even self-propelled since we don't actually move the robot via throttle but via carrying it around to simulate movement. 

We need to use a method to calculate the velocity of the robot. This can be done via taking the Zed2's IMU via the topic `/zed/zed_node/imu/data` for `IMU` messages containing the linear acceleration and angular velocity. But this is inaccurate. Use the Pose from `zed/zed_node/pose`.

#### What to use OGM from RTabmap for:
Either use OGM to:
- Ignore points where occupancy grid states is occupied is placed and don't put it into KD Tree. There is no reason to check Points at an (x,y) position that is occupied.
- For initializing valid particle positions (and checking is particles are just wrong position) - if a particle spawns in an occupied grid - resample it. If particle moves into an occupied grid - penalize its weight/set possibility to very low/zero
- Simplify calculations (instead of matching all points in the pointcloud of Rtabmap) - During localization, project the robot’s point cloud onto the 2D plane and compare it to the occupancy grid instead of directly matching 3D points. 
    - This abstracts away small positional shifts in obstacles (e.g a cone being moved just slightly forwards), as the grid represents the environment at a coarser level instead of point by point.

### Sub-Problem 2: Custom YOLO Publisher for Position of objects in world with 
The second sub problem is about the YOLO model used for image classification. 

Given an image, and the bounding box of an object, and a `CameraInfo` containing the camera's intrinsic position, how can we get the position (and uncertainty) of the object in camera frame (and then subsequently world frame). The nice thing is that Zed2 already gives us these values as an `ObjectStamped` from its image detection models. 

The only thing to use a different YOLO model is really to load in a custom ONNX file. We can steal these pretrained YOLO model parameters from Ultralytics, thereby bypassing needing to train models ourselves. Also bypasses needing to implement a custom publisher.

The output of this problem is preferably a position of the object in the camera frame or world frame. In Zed2's case, its all in world frame already.

Results from our testing show that the `Medium` model does relatively well at detecting humans and cars without having too high load on the Nvidia Jetson.

## Assumptions
### Gaussian Distribution
Assume Gaussian distribution - this makes it easy to track the uncertainty of a guess of a position of an object which we can use to determine how trustworthy a position recording of an object in world space is:
1) Dynamics model (linear and angular velocity) due to Odometry drifts - and also given by `PosewithCovariance` object from Zed2 Visual Inertial Odometry 
    - This returns as a 6x6 covariance matrix for (x, y, z, rotation about x axis, rotation about y axis, rotation about z axis).
    - Note: Depends on what you need, might just throw away all values except the 3x3 covariance matrix for (x, y, z).
2) Measurement model positions of objects in world frame given by Yolo Models of Zed2 - `ObjectStamped` and `Object` from the Zed2 YOLO gives us a 6-value covariance
    - This 6-value covariance for (x, y, z) position can be re-expanded out to its original 3x3 form for comparison
3) Guessestimate of Position of car - since its based on many particles each acting as a possible hypothesis for where the car is and its orientation
    - The covariance of the (x, y, z) position of the car position can be calculated via the sample variance formula (with bessel's correction): $s^2=\frac{\sum(x_i-\bar x)^2}{n-1}$
- To use the covariance matrix - take the trace of the covariance matrix. We can also use the inverse of the determinant (since it represents volume), but all we really care about in practice is the *variance* of the (x, y, z) position. This represents how trustworthy a position estimate of an object classified is.
- Gaussian distribution is nice as well, as it allows use to linearly add up the means and variances as we calculate particle positions using the dynamics and update it via observations

In practice, the way we do this tracking of whether to trust an object is based on variance - it can be a `Dictionary` or a `csv` of a running list of variances calculated by the trace(covariance matrix). If a new observation has a lower `variance`, This updates the entry in the CSV/dictionary.

| Object | Position (x, y, z) | Variance (trace of covariance matrix) |
| ---  | --- | ---|
| A | (1, 1, 0.5) | 15|
| A' | (1.5, 1.5, 0.75)| 5|

A' in this case is lower in variance, so it takes precedence in the CSV, we overwrite the entry.

### Starting position of car fixed in world frame based on the known map
The car always starts at a fixed position in the map at the start of the race track, and has a fixed goal. The world frame is known and fixed against the known map recorded (rather than the Odom position of the robot).

### Dynamics model
The car is best represented by a non-holonomic model.

By right since the car has a turning radius, it should be an Ackermann model. However, it might be a bit messy to get the update equations for the robot and particles. It is also hard to measure the turn radius of this car due to the inconsistency of the pull strength of the servo, so its turn radius is never really the same.

Even more advanced real world models have other parameters/constants such as friction, mass, but its overkill given how particle filter just needs an estimate.

Our pick is the easiest non-holonomic model out there, given our car has a linear and angular velocity. Our car can only move forward and backward via the throttle, and has no sideways motion. It may be simple but since particle filters work by estimation, its good enough.

$$
\begin{aligned}
x_{t+1} &= x_t + v \cdot \cos(\theta) \cdot \Delta t \\
y_{t+1} &= y_t + v \cdot \sin(\theta) \cdot \Delta t \\
\theta_{t+1} &= \theta_t + \omega \cdot \Delta t
\end{aligned}
$$

> Problem: Throttle being 1.0 (max value) might not actually be 1.0m/s in practice. This means to update the particles with the dynamics might not be as accurate as we expect... Either we record the speed at 1.0m/s as a constant. Or we use some IMU as a fallback. 

Here is the thing. The problem with the throttle not being 1.0m/s and stuff is not a big deal. Zed2 already IMU Fusion its Visual feed with Inertia and publishes it onto `/zed/zed_node/pose` as a `PosewithCovariance`. This should be accurate enough in the local scale. Hence to any particles in our system, as our car moves, we can simply store a previous_pose state, and then compare against the current pose state, and take the Translation and Rotation quaternion from timestep (t-1) to timestep t. Then apply these affine transforms to each particle (considering their gaussian noise e.g using `np.random.normal` to sample some fixed noisy movement based on the returned covariance from zed2). We do not explicity ever need to know what model our car represents, just observe and apply.

### Measurement model: Position of points in pointcloud
Rather than use the objects detected as landmarks (we opt for this in our report), the fact that we have a point cloud map means that we can figure out what points are seen in a pointcloud. This can then be used to update our particle's possibly of hypothesis.

## How to test our model? For the simplified version using Rtabmap capabilities to localize the LapBot

Technically, we can test our model anywhere, even not on the racetrack or without a battery to run the motors. 

All we need to do is to set up the objects, carry the robot around to build a pointcloud map using Rtabmap. This will becomes the known map for testing. In our case, we did this in the lab by placing down the objects in a position and pushing our car back and forth!

Now, we can run rtabmap with the database selected, start our node logic to deal with getting objects from zed2, then push it around the world encountering obstacles.

## ROS2 Topics for Zed2 camera
`/zed/zed_node/pose` - `Pose` Camera pose referred to Map frame (complete data fusion applied) - Can be used to get current position in space
`/zed/zed_node/odom`: Odometry pose referred to odometry frame (only visual odometry is applied for ZED, visual-inertial for ZED Mini)
`/cmd_vel` - `Twist` - To control the robot - max throttle is 1.0 (but it doesn't mean its 1.0m/s)
`/zed/zed_node/point_cloud/cloud_registered` - Get `PointCloud2` from Robot camera from Zed2 - recommended not to open RViz2 due to heavy processing on Jetson devices. This `PointCloud2` messages contain the Points that are being seen right now live from the camera. Ridiculously high fidelity with very dense PointCloud compared to Rtabmap.
`/rtabmap/cloud_map` - Get `PointCloud2` from Robot from RTabMap, loading from the recorded map database. This is published by Rtabmap when it runs (you can run it without `rtabviz` just for the publisher), and act as the point cloud representing the known map/pre-recorded pointcloud map. 


```python
# To read a pointcloud2 message 
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

def callback_pointcloud(data):
    assert isinstance(data, PointCloud2)
    gen = point_cloud2.read_points(data)
    print type(gen)
    for p in gen:
      print p  # type depends on your data type, first three entries are probably x,y,z
```


## Bonus Problem: Making SLAM

The problem is stil a manually driven car. The only difference is the lack of a map, instead the car drives around, builds its map, then figures out the position of objects in the map against a fixed world point like an Odom position. It might employ loop closure techniques to fix Odometry.

