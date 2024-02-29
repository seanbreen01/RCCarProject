##This will be main final driving control, and detection logic script when complete
import sys
import threading
import cv2
import cv2.aruco as aruco
import smbus
import time
import numpy as np

# All setup code here

# Corner slope variables
slopeLeft = 0
slopeRight = 0

# Video input setup, will alter based on testing results
xres = 1280	
yres = 720
framerate = 60

# Aruco marker detection variable
count = 0

# Nvidia Jetson Nano i2c Bus 0
bus = smbus.SMBus(0)
# This is the address we setup in the Arduino script
address = 0x40

# Aruco setup
    #--> dictionary of markers to be used, how will we be leveraging them
    #--> i.e. for corner type identification (this type of Aruco==hairpin, etc.)
    #--> or for position synchronization around track?
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
#	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
#	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
#	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
#	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
    
# Load the predefined dictionary
aruco_dict = aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_50"])
# Initialize the detector parameters using default values
parameters = aruco.DetectorParameters_create()

# CUDA setup stuff
gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, -1, (25,25), 5)
#other filters etc. here 
image_gpu = cv2.cuda_GpuMat() 


# I2C communications to Arduino UNO
def writeToArduino(valueToWrite):
    bytesToWrite = []
    for value in valueToWrite:
        # Split each integer into two bytes (high byte and low byte)
        highByte = (value >> 8) & 0xFF
        lowByte = value & 0xFF
        bytesToWrite.extend([highByte, lowByte])

    # Send the byte array
    print('Bytes: ')    #For debug
    print(bytesToWrite)
    bus.write_i2c_block_data(address, 0, bytesToWrite)

def readFromArduino():
    number = bus.read_byte(address)
    # number = bus.read_byte_data(address, 1)

    #TODO update to read from bus in correct format
    return number




# Video (GStreamer pipeline) setup
    #--> need to idetify framerate, resolution, etc. --> real world testing to inform, can be gradually refined
def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int){xres}, height=(int){yres}, format=(string)NV12, framerate=(fraction){frames}/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int){xres}, height=(int){yres}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink")

## have here as separate function for now, but will be integrated into main loop??
def videoCapture():
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    return cap

def processingPipeline(frame):
    image_gpu.upload(frame)
    gray_gpu = cv2.cuda.cvtColor(image_gpu, cv2.COLOR_BGR2GRAY)

    returnedImage = gray_gpu.download()

    # Don't want to detect Aruco markers every frame
    if count % 10 == 0:
        
        corners, ids, rejected_img_points = aruco.detectMarkers(returnedImage, aruco_dict, parameters=parameters)
        if ids is not None and len(ids) > 0:
            print('Aruco identified:', + ids)
            #check aruco marker against stored sequence if this exists?

        # For debugging if needed
        # print("Corners" + str(corners))
        # print("ids" + str(ids))
        # print("rejected" + str(rejected_img_points))
        count = 0
    count += 1
    

    # Steps ultimately leading to hough lines / canny edge / whatever line detection method is chosen

    # TODO
    # If line(s) detected, can go to slope detection step
    # If not, log no lines detected, and continue to next frame
    # Check also if no lines detected has happpened before
    # If no detections for X frames, send control commands to Arduino to stop car, try to regain position on track (recovery protocol)

    slopeDetection(returnedImage)

# Function for line slope detection
def slopeDetection(processedFrameResults):
    print("Slope detection")
    # TODO Tune parameters here
    rho = 1                  # distance resolution in pixels of the Hough grid
    theta = np.pi/180        # angular resolution in radians of the Hough grid
    threshold = 1            # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 20        # minimum number of pixels making up a line
    max_line_gap = 15        # maximum gap in pixels between connectable line segments


    lines = cv2.HoughLinesP(processedFrameResults, rho, theta, threshold, min_line_len, max_line_gap) 

    leftPointsX = []
    leftPointsY = []
    rightPointsX = []
    rightPointsY = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            m = (y1 - y2)/(x1 - x2) # slope
            
            # TODO tune values of these so appropriate to left and right respectively
            if m < -0.55: #and m > -5:
                leftPointsX.append(x1)
                leftPointsY.append(y1)
                leftPointsX.append(x2)
                leftPointsY.append(y2)
            elif m > 0.55: #and m < 5:
                rightPointsX.append(x1)
                rightPointsY.append(y1)
                rightPointsX.append(x2)
                rightPointsY.append(y2)


    # Lines if any detected, progress to determine what type of corners they represent
    cornerTypeDetection(leftPointsX, leftPointsY, rightPointsX, rightPointsY)

# Function for corner type detection --> is hairpin, trigger these control responses
def cornerTypeDetection(leftPointsX, leftPointsY, rightPointsX, rightPointsY):
    print("Corner type detection")

    # TODO calculate slope using left and right x and y points
    slopeLeft = leftPointsX / leftPointsY

    slopeRight = rightPointsX / rightPointsY

    # TODO add all corner conditions and tune slope values
    # Straight ahead
    if slopeLeft >= 100 and slopeRight >= 100:
        # commands to steering and motor to continue straight ahead, maybe increase speed?
        print("Straight ahead")
        #TODO if mapping track
        # --> save corner type to array, & corresponding timestamp/
        cornerType = "Left Hairppin" # Have this be a key for a dictionary of corner types and their associated control commands
        

    elif slopeLeft > 0.5 and slopeRight < -0.5:
        print("Hairpin detected")
        #--> send control commands to Arduino to slow down, turn, etc.

    elif slopeLeft > 0.5 and slopeRight > 0.5:
        print("Left curve detected")
        #--> send control commands to Arduino to slow down, turn, etc.
    


    sendControlCommands()

# Function to send control commands
    #--> non-blocking if series of commands is needed in eventual 'racing-line' following implementation
    #--> format of sent commands already known
def sendControlCommands():
    writeToArduino([0, 80, 1000])   #steering control
    writeToArduino([1, 1600, 1000]) #motor control



# Function for Aruco detection and storage to "list"/array etc. so when at full speed can do detections and say:
# "passed aruco 3, then 4, then 5 etc. this is in correct (mapped) order" 
# --> above needs more detailed consideration

# Function for frame processing
    # --> Needs to be rolling average // way to isolate bad frames due to blur, no detections etc. so system doesn't freak out and stop prematurely
    #--v avoids issue of bad detections, blurred frames, etc. 


# Function to generate control commands? --> based on line slope


# Function for automated recovery should off track event happen for any reason
def automatedRecovery():
    print("Automated recovery protocol initiated")
    #--> pre-written series of inputs until line(s) detected again and Aruco markers in sequence expected
    #--R if 2 aruco markers are reverse of what should be seen, navigating track wrong direction, turn and re-continue in correct way

def main():
    print('Main loop')
    cap = videoCapture()

    if cap.isOpened():
        try:
            while True:
                _, frame = cap.read()  # Read a frame from the camera

                #TODO Immeadiately remove excess data from frame, i.e. whatever camera is capturing of the sky / horizon / static parts of car itself
                # Then pass this frame in to processing 
                processed_frame_results = processingPipeline(frame)
                # ^ Will return arrays/lists with points that equate to detected lines

                keycode = cv2.waitKey(10) & 0xFF

                if keycode == 27 or keycode == ord('q'): #allow user to quit gracefully
                    break
        finally:  
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("cap.isOpened() Error")




# Main process control loop
if __name__ == "__main__":
    print("Drive initialised")
    
    main()
    #FrameIngest
    #Slope outputs
    #Control decisions based on slope
    #Send control decisions to Arduino
    #Receive ACK --> strictly necessary or nice to have? might just be more processing and cross-talk that doesn't necessarily add a whole pile of useful functionality?
    #begin again