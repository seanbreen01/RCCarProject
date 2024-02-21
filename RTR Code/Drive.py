##This will be main final driving control, and detection logic script when complete
# For now: Stub code

import cv2
import numpy as np
import time
import smbus

# setup code for I2C communications
    #--> send and receive
# CUDA setup stuff here also
# Video (GStreamer pipeline) setup
    #--> need to idetify framerate, resolution, etc. --> real world testing to inform, can be gradually refined
# Aruco setup?
    #--> dictionary of markers to be used, how will we be leveraging them
    #--> i.e. for corner type identification (this type of Aruco==hairpin, etc.)
    #--> or for position synchronization around track?

# Function for corner type detection --> is hairpin, trigger these control responses
# Function for line slope detection
# Function for Aruco detection and storage to "list"/array etc. so when at full speed can do detections and say:
# "passed aruco 3, then 4, then 5 etc. this is in correct (mapped) order" 
# --> above needs more detailed consideration

# Function for frame processing
    # --> Needs to be rolling average // way to isolate bad frames due to blur, no detections etc. so system doesn't freak out and stop prematurely
    #--v avoids issue of bad detections, blurred frames, etc. 

# Function to send control commands
    #--> non-blocking if series of commands is needed in eventual 'racing-line' following implementation
    #--> format of sent commands already known
# Function to generate control commands? --> based on line slope

# Function for automated recovery should off track event happen for any reason
    #--> pre-written series of inputs until line(s) detected again and Aruco markers in sequence expected
    #--R if 2 aruco markers are reverse of what should be seen, navigating track wrong direction, turn and re-continue in correct way




# Main process control loop
if __name__ == "__main__":
    print("Drive initialised")
    #FrameIngest
    #Slope outputs
    #Control decisions based on slope
    #Send control decisions to Arduino
    #Receive ACK --> strictly necessary or nice to have? might just be more processing and cross-talk that doesn't necessarily add a whole pile of useful functionality?
    #begin again