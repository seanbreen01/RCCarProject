import cv2
import cv2.aruco as aruco
import numpy as np
import time

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

# Below works for opencv versions 4.7.x and beyond

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# aruco_dict = aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_50"])
# # Initialize the detector parameters using default values
# parameters = aruco.DetectorParameters_create()

def gstreamer_pipeline(
    capture_width=1280, #lowered from 1920x1080 for improved speed
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=60,   #increased from 30 again to improve speed
    flip_method=2,
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

# TODO Benchmark permutations
startTime = time.time()

# define a region of interest mask
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

def imageProcessing(frame):
    # Cut down image size to improve speed
    # TODO set size appropriately
    # frame = frame[100:720, 80:1200]

    DEBUG = False

    # TODO Define region of interest approrpriate to camera angle
    # Note: order of vertices is important to get the correct mask shape
    # region_of_interest_vertices = np.array([[80, 720],  [80, 250], [1200, 250],[1200, 720]], dtype=np.int32)

    # ROI defined as trapezium for 720 video
    region_of_interest_vertices = np.array([[0, 720], [30, 150], [1250, 150], [1280, 720]], dtype=np.int32)

    roi_image = region_of_interest(frame, [region_of_interest_vertices])
    # cv2.imshow('ROI', roi_image)

    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # Aruco detection
    # corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    corners, ids, rejected_img_points = detector.detectMarkers(gray)

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

    # maybe need to blur here?
    # TODO tune if needed
    kernel_size = (9, 9)
    sigmaX = 3
    sigmaY = 1
    gray = cv2.GaussianBlur(gray, ksize=kernel_size, sigmaX=sigmaX, sigmaY=sigmaY)

    # Detect edges
    canny_low_thresh = 40
    canny_high_thresh = 120

    edges = cv2.Canny(gray, canny_low_thresh, canny_high_thresh)

    cv2.imshow('Edges', edges)

    # TODO Declare once only and tune as needed 
    # Hough transform
    rho = 1                  # distance resolution in pixels of the Hough grid
    theta = np.pi/180        # angular resolution in radians of the Hough grid
    threshold = 1            # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20     # minimum number of pixels making up a line
    max_line_gap = 15        # maximum gap in pixels between connectable line segments

    houghLines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=10)
    # TODO find out about extracting only the longest lines and if this would be representative of the lane lines?
    # houghLines = np.sort(houghLines)
    # print(houghLines)
    # print("Length:", len(houghLines))


    if houghLines is not None:
        line_img = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        leftLaneSlope, rightLaneSlope = laneSplit(line_img, houghLines)

        #cv2.imshow('Hough', line_img)

        combined = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
        # cv2.imshow('Combined', combined)
    else:
        combined = frame
    
    #Do some more with frame, etc. etc. 


    


    # Will actually return a command to send to the Arduino and affect some change in its movement
    return combined

def main():
    video_path = 'Videos/kitchenadjcamerayesaruco720_30.avi' 

    cap = cv2.VideoCapture(video_path)

    window_title = "RCCar Video Output Debug Feed"

    if cap.isOpened():
        
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            

            while True:
                _, frame = cap.read()  # Read a frame from the camera
                #logic here 
                processed_frame = imageProcessing(frame)

                

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, processed_frame)
                    #Not needed if not showing output frames in actual implementation
                    

                else:
                    break
                keycode = cv2.waitKey(10) & 0xFF

                if keycode == 27 or keycode == ord('q'): #allow user to quit gracefully
                    break
        
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Error")

if __name__ == '__main__':
    main()