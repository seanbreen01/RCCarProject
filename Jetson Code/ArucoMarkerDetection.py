import cv2
import cv2.aruco as aruco
import sys

def find_aruco_markers(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Could not open or find the image")
        sys.exit()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the predefined dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

    # Initialize the detector parameters using default values
    parameters = aruco.DetectorParameters_create()

    # Detect the markers
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Check if markers were found
    if ids is not None and len(ids) > 0:
        print("Marker found")
    else:
        print("Nada")

# Replace 'path_to_image.jpg' with the path to your input image
find_aruco_markers('path_to_image.jpg')
