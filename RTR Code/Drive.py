##This will be main final driving control, and detection logic script when complete
import sys
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
framerate = 30

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
# TODO add all necessary functions and filters etc. here
gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, -1, ksize=(9,9), sigmaX=3, sigmaY=1) #TODO 8UC3 - what is this?

cannyEdgeDetector = cv2.cuda.createCannyEdgeDetector(low_thresh=40, high_thresh=120)

houghLinesDetector = cv2.cuda.createHoughLinesDetector(rho=1, theta=np.pi/180, threshold=50, doSort=True, maxLineGap=10, minLineLength=50) # TODO what is doSort and tune further

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
    print('Bytes: ')    #TODO add flag for debug
    print(bytesToWrite)
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
# TODO use CUDA for this? i.e. bitwise_and operation
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
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

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
    y2 = int((maxY/4)) + 60  # note: hardcoded, sets the length of the line to half the image height + 60 pixels, original value was div by 2
    x2 = int((y2 - b)/m)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Helper function - split the detected lines into left and right lines
def laneSplit(img, lines, color=[0, 255, 0], thickness=20):

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
            
            # TODO tune these   
            if m < -0.5:
                leftPointsX.append(x1)
                leftPointsY.append(y1)
                leftPointsX.append(x2)
                leftPointsY.append(y2)
                mLeft.append(m)
                
            elif m > 0.1 and m < 2:
                rightPointsX.append(x1)
                rightPointsY.append(y1)
                rightPointsX.append(x2)
                rightPointsY.append(y2)
                mRight.append(m)
    
    # TODO remove from non debugging code, or add control flag
    drawLine(img, leftPointsX, leftPointsY, color, thickness)
        
    drawLine(img, rightPointsX, rightPointsY, color, thickness)

    # TODO in progress here, my additions
    # TODO this slope calculation is not working as intended
    # TOFIX needs to be proper m = (y2 - y1)/(x2 - x1) calculation as above. 
    # slopeLeftPoints = [x/y for x,y in zip(leftPointsX, leftPointsY)]
    # slopeLeftPoints = sum(slopeLeftPoints)/len(slopeLeftPoints)

    
    avg_left_slope = np.mean(mLeft)
    avg_right_slope = np.mean(mRight)

    # if avg_left_slope < -4:
    #     cv2.waitKey(0)

   # print("Averge slope left line points", avg_left_slope)

    return avg_left_slope, avg_right_slope

def processingPipeline(frame):

    DEBUG = False

    # ROI defined as trapezium for 720 video
    region_of_interest_vertices = np.array([[0, 720], [30, 150], [1250, 150], [1280, 720]], dtype=np.int32)

    roi_image = region_of_interest(frame, [region_of_interest_vertices])
    # cv2.imshow('ROI', roi_image)

    image_gpu.upload(roi_image)

    gray_gpu = cv2.cuda.cvtColor(image_gpu, cv2.COLOR_BGR2GRAY)

    

    blurred_gpu = gaussian_filter.apply(gray_gpu)

    cannyEdgesDetected_gpu = cannyEdgeDetector.detect(blurred_gpu)

    houghlines_gpu = houghLinesDetector.detect(cannyEdgesDetected_gpu)

    houghLines = houghlines_gpu.download()

    if houghLines is not None:
        # TODO edges refer to canny edges, need to download or see if move to CPU ops should occur by this point
        line_img = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        leftLaneSlope, rightLaneSlope = laneSplit(line_img, houghLines)

        #cv2.imshow('Hough', line_img)

        combined = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
        # cv2.imshow('Combined', combined)
    else:
        combined = frame

    returnedImage = gray_gpu.download()

    #TODO how often
    # Don't want to detect Aruco markers every frame
    if count % 5 == 0:
        
        corners, ids, rejected_img_points = aruco.detectMarkers(returnedImage, aruco_dict, parameters=parameters)
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

            cv2.imshow('Detected Markers', frame)
        elif DEBUG == True:
            print("No marker present in frame")

                #TODO check aruco marker against stored sequence if this exists?

        
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

    # TODO add all corner conditions and tune slope values, obviously 100 there now is way off
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
    pipeline = gstreamer_pipeline(capture_width=xres, capture_height=yres, display_width=xres, display_height=yres, framerate=framerate, flip_method=2)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

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