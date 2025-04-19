# For the Lane Following Bot

The control system is always a PID controller - the PID parameters will be the same regardless of which implementation is chosen.

## Method 1: Uses the dumb dumb OpenCV Blob tracking way

This one can be built directly as long as OpenCV and CV bridge (for ROS2 Image messages to OpenCV images conversion) is available on the 1/10th car.

This has low resource overhead though.

Downside is that its pixel based - the car doesn't really know exactly how far out it is from the left and right lane. 

Another downside is that it fail in scenarios where there are multiple candidates for connected components/blobs in the image . Our team does take some precautions by caching the position of centroids of left and right detected blobs and keeping a threshold on the max distance away from old values a blob can be. But this may still cause it to lock onto the wrong blobs.

## Method 2: LaneNet DL

The nice thing is that someone has wrapped LaneNet Model into a ROS package: https://github.com/AbangLZU/LaneNetRos

This uses Deep Neural Network based around image segmentation to guess the lanes. It is a lot smarter than just purely counting pixels and taking their centers.

After installing the LaneNetRos based on their instructions and downloading the pretrained weights, a ROS Python Node has been written that imports the LaneNetROS package and uses it to pick up lanes.
