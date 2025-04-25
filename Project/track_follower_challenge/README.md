# Track follower: Keeping in line

This implements a lane detector + PID controller to keep our car in lane.

The control system is always a PID controller - the PID parameters will be the same regardless of which implementation is chosen.


## To run Part 1
1. Put folder into your workspace under `/<ros2_ws>/src/`

2. Colcon build the new package: `colon build --symlink-install --packages-select track_follower_challenge`
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

5. Start the Zed2 camera: `ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2`

6. Start the node to control the car: `ros2 run track_follower_challenge track_follower`
    - If left and right lane is detected, the OpenCV window showing the hough lines will pop up
    - You can also open `rviz` where we send the debug frames to `/debug/color/masked_image` - these contain the same thing as the OpenCV for recording
    - Make sure the car is in autonomous mode by pressing the joycon O 


## Methods of implementation 
### Method 1: Uses the dumb dumb OpenCV Blob tracking way

This one can be built directly as long as OpenCV and CV bridge (for ROS2 Image messages to OpenCV images conversion) is available on the 1/10th car.

This has low resource overhead though.

Downside is that its pixel based - the car doesn't really know exactly how far out it is from the left and right lane. 

Another downside is that it fail in scenarios where there are multiple candidates for connected components/blobs in the image . Our team does take some precautions by caching the position of centroids of left and right detected blobs and keeping a threshold on the max distance away from old values a blob can be. But this may still cause it to lock onto the wrong blobs.

### Method 2: LaneNet DL

The nice thing is that someone has wrapped LaneNet Model into a ROS package: https://github.com/AbangLZU/LaneNetRos
The bad thing is that its not directly usable - the publisher ROS node that its written in is the older ROS 1 API using `rospy`. 

To have it work with `rclpy` in ROS2, we can basically hijack the internal libraries in the repo, and rewrite the code for the node to directly call the internal tensorflow code.

This uses Deep Neural Network based around image segmentation to guess the lanes. It is a lot smarter than just purely counting pixels and taking their centers.

After installing the LaneNetRos based on their instructions and downloading the pretrained weights, a ROS Python Node has been written that imports the LaneNetROS package and uses it to pick up lanes.

## Problems faced

### LaneNet is impossible to run
Memory stuff, CUDA setup makes the versions of LaneNet 

### Car is dummy thicc on the battery - never goes straight only gay
Due to the weight of battery, it drives with a right bias. This meant that the PID controller would struggle to pull it back to the middle of the lane.

Our team bypassed this problem by adding a linear bias to the error (measured in pixel from the x coords of vanishing point). This bias was manually tested until we found a decent value that would let us run 100m on straights without touching the line.

### OpenCV is annoying

Figuring out the pipeline and hyperparameters of preprocessing a camera image probably took the most time of the project. We tried various methods of smoothing (e.g via adding noise), various parameters of Canny Line detector, increasing contrast of image, hough line clustering, gradient thresholds to really get a set of very good estimates of the left and right lane. Without good preprocessing, calculating the error for the PID controller is impossible, as we need a vanishing point to deal where is the middle of the lane (and to chase).

### Track is borked

Some parts of the track has a line that is faded. 
