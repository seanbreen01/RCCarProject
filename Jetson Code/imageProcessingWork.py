import cv2
import cv2.aruco as aruco
import numpy as np
import time

print("Setup initialised")

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

gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, -1, (25,25), 5)
image_gpu = cv2.cuda_GpuMat()   #declaring CUDA object into which we can pass images for processing with onboard GPU

#TODO add prints for variable settings, useful for debugging in real world testing especially 
print("Setup complete")

#TODO adjust parameters of GStreamer pipeline based on testing results for optimal resolution and framerate
def gstreamer_pipeline(
    capture_width=1280, #lowered from 1920x1080 for improved speed
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=60,   #increased from 30 again to improve speed
    flip_method=6,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

startTime = time.time() # TODO put this where it is appropriate

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
window_title = "RCCar Video Output Debug Feed"

def imageProcessing(frame):
    image_gpu.upload(frame)  

    gray_gpu = cv2.cuda.cvtColor(image_gpu, cv2.COLOR_BGR2GRAY)
    #TODO remove from final build 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #for temporoary debugging

    returnedImage = gray_gpu.download()

    # Detect the markers
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # For debugging if needed
    # print("Corners" + str(corners))
    # print("ids" + str(ids))
    # print("rejected" + str(rejected_img_points))

    if ids is not None and len(ids) > 0:
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
    else:
        print("No marker present in frame")

    
    
    #Do some more with frame, etc. etc. 


    

    


    # Will actually return values to send to the Arduino and affect some change in its movement
    return gray

def loopy():

    if cap.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

            while True:
                _, frame = cap.read()  # Read a frame from the camera
                #logic here 
                processed_frame = imageProcessing(frame)

                

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    #cv2.imshow(window_title, processed_frame)
                    #Not needed if not showing output frames in actual implementation
                    print('Output window for Debugging')

                else:
                    break
                keycode = cv2.waitKey(10) & 0xFF

                if keycode == 27 or keycode == ord('q'): #allow user to quit gracefully
                    break
        finally:  
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Error")