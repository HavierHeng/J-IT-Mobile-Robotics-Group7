# Track follower: Keeping in line

This implements a lane detector + PID controller to keep our car in lane.

# Problems faced

## Car is dummy thicc on the battery - never goes straight only gay
Due to the weight of battery, it drives with a right bias. This meant that the PID controller would struggle to pull it back to the middle of the lane.

Our team bypassed this problem by adding a linear bias to the error (measured in pixel from the x coords of vanishing point). This bias was manually tested until we found a decent value that would let us run 100m on straights without touching the line.

## OpenCV is annoying

## Track is borked

Some parts of the track has a line that is not very coloured
