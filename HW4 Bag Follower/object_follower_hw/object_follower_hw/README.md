# To run HW4 Part 2


This assumes the `ros_ws` workspace from the earlier practical is already built.

1. Copy the obj_det_visualizer in the HW4 folder into the `~/ros2_ws/src/` as a new package

2. cd into the `~/ros2_ws` workspace.

3. Colcon build the new package: `colon build --symlink-install --packages-select object_follower_hw`
    - Recommendation: Avoid building all packages, as it may include extra things already built - Zed2 takes quite long to do so.

4. Source the workspace overlay: `source ~/ros_ws/install/local_setup.bash`

5. Start remote control of RC Car
    - open terminal: `remote_control`
    - press reset button of arduino zero board
    - Option 1:
        - open new terminal:`ros2 run joy joy_node`
        - open new terminal type: `ros2 run car_control car_control`

    - Option 2:
        - Run this command in terminal: `ros2 launch car_control car_control.launch.py`
  
6. For bounding box visualizer just use the one given in the practical and start rviz. In new terminals for each command:
    - Start the Zed2 camera: `ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2`
    - Start the given visualizer for bounding boxes: `ros2 run obj_det_visualizer obj_visualizer`
    - Start RViz2, you should see a camera view and world view with marker representing the 3D bounding box from Zed2 object detection: `rviz2`

7. Start the node to control the car: `ros2 run object_follower_hw straight_follower`
    - Make sure the car is in autonomous mode by pressing the joycon

The normal version of the node runs this logic in a nutshell, it only moves straight via bang bang control of linear x velocity:

![Drive Straight and stop at bag](./hw4_diagram.png)
