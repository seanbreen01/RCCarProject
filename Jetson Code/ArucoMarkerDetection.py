import cv2
import cv2.aruco as aruco
import sys
import numpy as np

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

def find_aruco_markers(frame):
    # Load the image
    # image = cv2.imread(image_path)

    if frame is None:
        print("Could not open or find the image")
        sys.exit()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Below works for opencv versions 4.7.x and beyond, commented version should work for those before that
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected_img_points = detector.detectMarkers(gray)
    print("Corners" + str(corners))
    print("ids" + str(ids))
    print("rejected" + str(rejected_img_points))


    # Load the predefined dictionary
    # aruco_dict = aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_50"])
    # Initialize the detector parameters using default values
    # parameters = aruco.DetectorParameters_create()
    # Detect the markers
    # corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Check if markers were found
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
        cv2.waitKey(0)


    else:
        print("No marker present in frame")

# Replace 'path_to_image.jpg' with the path to your input image
#find_aruco_markers('./Markers/Marker_1.png')

cap = cv2.VideoCapture(0)	
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        try:
            cv2.imshow('frame', frame)
            find_aruco_markers(frame)
            
        finally:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

# find_aruco_markers("./image.png")
