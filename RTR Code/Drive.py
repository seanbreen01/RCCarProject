##This will be main final driving control, and detection logic script when complete
import sys
import cv2
import cv2.aruco as aruco
import smbus
import time
import numpy as np

# All setup code here
# Debug variable
DEBUG = False

# Video input setup
xres = 1280	
yres = 720
framerate = 30

# Control theory varaible definitions
maxSpeed = int(input("Set max speed 1600 - 2000: "))

cornerType = "straight"

# leftLaneSlope = 0
# rightLaneSlope = 0

left_lane_slopes = []
right_lane_slopes = []

window_size = 5

cornerTypeCounter = 0

corner_dict_steering = {
    "straight": [0,75,250],
    "gentleLeft": [0, 90, 250],
    "gentleRight": [0, 65, 250],
    "rightTrim": [0, 70, 250],
    "leftTrim": [0, 80, 250]
    }

corner_dict_motor = {
    "straight": [1,maxSpeed,250],
    "gentleLeft": [1, maxSpeed, 250],
    "gentleRight": [1, maxSpeed, 250],
    "rightTrim": [1, maxSpeed, 250],
    "leftTrim": [1, maxSpeed, 250]
    }

i2cErrorCounter = 0

# Aruco marker detection variable
counter = 0

# Nvidia Jetson Nano i2c Bus 0
bus = smbus.SMBus(0)
# address  setup in the Arduino script
address = 0x40

# Aruco setup
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
# TODO add all necessary functions and filters etc. here
gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, -1, ksize=(9,9), sigma1=9, sigma2=5) #defines source image as 8-bit single colour channel (grayscale, and -1 is destination the same)

cannyEdgeDetector = cv2.cuda.createCannyEdgeDetector(low_thresh=50, high_thresh=120)

region_of_interest_vertices = np.array([[0,80], [1280,80], [1280,720], [1240,720], [980,300],[300,300], [40,720],  [0,720]], dtype=np.int32)

#houghLinesDetector = cv2.cuda.createHoughLinesDetector(rho=1, theta=np.pi/180, threshold=50, doSort=True, maxLines=50) # TODO what is doSort, maxLines and tune further

#gpu object
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
    #print('Bytes: ')    #TODO add flag for debug
    #print(bytesToWrite)
    bus.write_i2c_block_data(address, 0, bytesToWrite)

def readFromArduino():
    number = bus.read_byte(address)
    # number = bus.read_byte_data(address, 1)

    #TODO update to read from bus in correct format
    return number


# Video (GStreamer pipeline) setup
    #--> need to identify framerate, resolution, etc. --> real world testing to inform, can be gradually refined
def gstreamer_pipeline(sensor_id=0, sensor_mode=3, capture_width=1280, capture_height=720, display_width=640, display_height=480, framerate=30, flip_method=2):
    return (
        f'nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! '
        f'video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, '
        f'format=(string)NV12, framerate=(fraction){framerate}/1 ! '
        f'nvvidconv flip-method={flip_method} ! '
        f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! '
        f'videoconvert ! '
        f'video/x-raw, format=(string)BGR ! appsink'
    )

# define a region of interest mask

# TODO strip unnecessary code from this function
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon defined by "vertices". 
    The rest of the image is set to black.
    """
    #define a blank mask
    mask = np.zeros_like(img)   
    
    #define a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    # masked_image = cv2.bitwise_and(img, mask)
    masked_image = cv2.cuda.bitwise_and(img, mask)
    return masked_image

#Ignored unless debug flag is true
def drawLine(img, x, y, color=[0, 255, 0], thickness=20):
    if len(x) == 0: 
        return
    
    lineParameters = np.polyfit(x, y, 1) 
    
    m = lineParameters[0]
    b = lineParameters[1]
    
    maxY = img.shape[0]
    maxX = img.shape[1]
    y1 = maxY
    x1 = int((y1 - b)/m)
    # Trims line draw height, used only in debug output image
    y2 = int((maxY/8)) + 60  # note: hardcoded, sets the length of the line to half the image height + 60 pixels, original value was div by 2
    x2 = int((y2 - b)/m)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Helper function - split the detected lines into left and right lines
def laneSplit(img, lines, color=[0, 255, 0], thickness=20):
    global DEBUG
    
    leftPointsX = []
    leftPointsY = []
    rightPointsX = []
    rightPointsY = []

    # TODO
    mLeft = []
    mRight = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            m = (y1 - y2)/(x1 - x2) # slope
            
            # TODO tune these m values, especial care needed
            if m < -0.1:
                leftPointsX.append(x1)
                leftPointsY.append(y1)
                leftPointsX.append(x2)
                leftPointsY.append(y2)
                mLeft.append(m)
                
            elif m > 0.1:
                rightPointsX.append(x1)
                rightPointsY.append(y1)
                rightPointsX.append(x2)
                rightPointsY.append(y2)
                mRight.append(m)
    
    if DEBUG == True:
        drawLine(img, leftPointsX, leftPointsY, color, thickness)
        drawLine(img, rightPointsX, rightPointsY, color, thickness)

    
    avg_left_slope = np.mean(mLeft)
    avg_right_slope = np.mean(mRight)

    #print('left', avg_left_slope)
    #print('right', avg_right_slope)

    return avg_left_slope, avg_right_slope

def processingPipeline(frame):

    global DEBUG
    global counter
    global cornerTypeCounter
    global average_left_slope
    global average_right_slope
    global window_size

    cornerTypeCounter += 1
 
    # 'Pants' shaped ROI
    # roi_image = region_of_interest(frame, [region_of_interest_vertices])
    # cv2.imshow('ROI', roi_image)


    image_gpu.upload(frame)

    gray_gpu = cv2.cuda.cvtColor(image_gpu, cv2.COLOR_BGR2GRAY)

    blurred_gpu = gaussian_filter.apply(gray_gpu)

    cannyEdgesDetected_gpu = cannyEdgeDetector.detect(blurred_gpu)

    #TODO how often
    # Don't want to detect Aruco markers every frame
    if counter % 10 == 0:
        print("Aruco Detection")
        corners, ids, rejected_img_points = aruco.detectMarkers(gray_gpu.download(), aruco_dict, parameters=parameters)
        if ids is not None and len(ids) > 0 and DEBUG == True:
            print("Marker found")
            for i in range(len(ids)):
                # Extract corner points
                corner = corners[i][0]
                
                # Draw bounding box
                rect = cv2.minAreaRect(corner)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

                text_position = (int(corner[0][0]), int(corner[0][1]))
                # Draw marker ID
                cv2.putText(frame, str(ids[i][0]), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            #cv2.imshow('Detected Markers', frame)
        elif DEBUG == True:
            print("No marker present in frame")

                #TODO check aruco marker against stored sequence if this exists?

        counter = 0
    counter += 1


    edges = cannyEdgesDetected_gpu.download()

    roi_image = region_of_interest(edges, [region_of_interest_vertices])
    #cv2.imshow('roi', roi_image)
    houghLines_cpu = cv2.HoughLinesP(roi_image, 1, np.pi/180, 50, maxLineGap=30)


    if houghLines_cpu is not None:
        # TODO can this definiton be moved outside of function?
        line_img = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        new_left_slope, new_right_slope = laneSplit(line_img, houghLines_cpu)

        # SLiding average window for slope values
        left_lane_slopes.append(new_left_slope)
        if len(left_lane_slopes) > window_size:
            left_lane_slopes.pop(0)
        average_left_slope = sum(left_lane_slopes) / len(left_lane_slopes)

        # Update right lane slopes
        right_lane_slopes.append(new_right_slope)
        if len(right_lane_slopes) > window_size:
            right_lane_slopes.pop(0)
        average_right_slope = sum(right_lane_slopes) / len(right_lane_slopes)

        # Used for debug purposes only
        #cv2.imshow('Hough', line_img)
        #combined = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
        #cv2.imshow('Combined', combined)
    else:
        #combined = frame
        average_left_slope = 0
        average_right_slope = 0

    # Steps ultimately leading to hough lines / canny edge / whatever line detection method is chosen

    # TODO
    # If line(s) detected, can go to slope detection step
    # If not, log no lines detected, and continue to next frame
    # Check also if no lines detected has happpened before
    # If no detections for X frames, send control commands to Arduino to stop car, try to regain position on track (recovery protocol)
        
    #TODO if in recovery procedure, don't detect corner type, create flag for this

    if cornerTypeCounter % 7 == 0 and average_left_slope is not None and average_right_slope is not None:
        cornerTypeDetection(average_left_slope, average_right_slope)
        cornerTypeCounter = 0
    


# Function for corner type detection --> is hairpin, trigger these control responses
def cornerTypeDetection(leftLaneSlope, rightLaneSlope):
    print("Corner type detection")
    centerMargin = 0.15

    global cornerType
    # TODO: explore if trying to minimise the difference between both lanes is the ideal path to take, handles every case in theroy but implementation may be difficult? 

    # TODO tune slope values, for additional corner types, and to ensure correct detection
    # Straight ahead
    
    if leftLaneSlope <= -0.1 and leftLaneSlope > -0.3 and rightLaneSlope >= 0.1 and rightLaneSlope < 0.3:
        # commands to steering and motor to continue straight ahead, maybe increase speed?
        print("Straight ahead")
        # TODO tune so its not 'overly' sensitive, adjust only when needed, not if its just not perfectly centered
        if leftLaneSlope > rightLaneSlope - centerMargin:
            # recenter on track, too far left
            print("shift right slightly")
            cornerType = "rightTrim"
        elif rightLaneSlope > leftLaneSlope + centerMargin:
            # recenter on track, too far right
            print("shift left slightly")
            cornerType = "leftTrim"

        cornerType = "straight" 

    elif leftLaneSlope < -0.3 and rightLaneSlope < 0.3:
        print("Gentle Left - both negative slope detected, [but left lane is steeper than right lane, so gentle right turn detected]???")
        cornerType = "gentleLeft"
    elif leftLaneSlope > -0.3 and rightLaneSlope > 0.3:
        print("Gentle Right - both positive slopes ") 
        cornerType = "gentleRight"
    # TODO need data on this to align values properly
    elif leftLaneSlope < -10 and rightLaneSlope < - 10:
        print("90 Degree Right - both strongly negative slopes")
            
    elif leftLaneSlope > 10 and rightLaneSlope > 10:
        print("90 Degree Left - both strongly positive slopes")    

    # TODO logic for hairpins seems dodgy at best, review with test footage
    elif leftLaneSlope < -10 and rightLaneSlope > 100:
        print("Right Entry Hairpin detected - left lane strongly negative, right lane almost flat horizontal line (close to infinite slope)")
        #--> send control commands to Arduino to slow down, turn, etc.
    elif leftLaneSlope > 100 and rightLaneSlope > 10:
        print("Left Entry Hairpin  - right lane strongly positive, left lane almost flat horizontal line (close to infinite slope)")
    

    elif np.isnan(leftLaneSlope) and rightLaneSlope is not None:
        print("No left lane, off track to left side of course (left lane incorrectly identified as right lane?)")
        cornerType = "gentleLeft"
    elif np.isnan(rightLaneSlope) and leftLaneSlope is not None:
        print("No right lane, off track to right side of course (right lane incorrectly identified as left lane?)")
        cornerType = "gentleRight"
    elif np.isnan(leftLaneSlope) and np.isnan(rightLaneSlope):
        print("Completely off track, engage recovery protocol")

    sendControlCommands(cornerType)

# Function to send control commands
    #--> non-blocking if series of commands is needed in eventual 'racing-line' following implementation
    #--> format of sent commands already known
def sendControlCommands(cornerType = None):
# TODO define time to hold variable rather than hardcoded 150ms values --> recipe for disaster
    global i2cErrorCounter

    try:
        writeToArduino(corner_dict_steering[cornerType])
        writeToArduino(corner_dict_motor[cornerType])
    except OSError:
        print("Command to Arudino failed to transmit")
        i2cErrorCounter += 1
        if i2cErrorCounter > 5:
            print("I2C failure detected, ending script here")
            if DEBUG == False:
                sys.exit()


# Function for Aruco detection and storage to "list"/array etc. so when at full speed can do detections and say:
# "passed aruco 3, then 4, then 5 etc. this is in correct (mapped) order" 
# --> above needs more detailed consideration


# Function for automated recovery should off track event happen for any reason
def automatedRecovery():
    print("Automated recovery protocol initiated")
    #--> pre-written series of inputs until line(s) detected again and Aruco markers in sequence expected
    #--R if 2 aruco markers are reverse of what should be seen, navigating track wrong direction, turn and re-continue in correct way

    #TODO call send commands function with predefined maneouver loop until track re-located and direction of rotation about it validated

def main():
    print('Main loop')
    pipeline = gstreamer_pipeline(capture_width=xres, capture_height=yres, display_width=xres, display_height=yres, framerate=framerate, flip_method=2)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    #video_path = 'Videos/fulltrack3720_30.avi' 
    #cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        try:
            while True:
                #timeStart = time.time()
                _, frame = cap.read()  # Read a frame from the camera

                processingPipeline(frame)
                #timeEnd = time.time()
                #print('FPS:', 1/(timeEnd-timeStart) )

                # ^ Will return arrays/lists with points that equate to detected lines

                #cv2.imshow("Debug", processed_frame_results)

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
    print("Debug is: ", DEBUG)
    main()