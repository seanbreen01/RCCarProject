import cv2
import cv2.aruco as aruco
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

def generate_aruco_markers(dictionary, marker_size, output_folder):
    for marker_id in range(len(dictionary.bytesList)):
        marker_image = aruco.drawMarker(dictionary, marker_id, marker_size, tag, 1)
        address = f"./Markers/Marker_{marker_id}.png"
        cv2.imwrite(address, marker_image)
        # cv2.imshow("tag", tag)
        # cv2.waitKey(0)

if __name__ == "__main__":
    aruco_dict = aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_50"])    #can change if needed
    tag = np.zeros((300, 300, 1), dtype="uint8")
    print('here')
    marker_size = 300  # size of the marker in pixels
    output_folder = "Aruco Markers"  # specify your output folder here
    generate_aruco_markers(aruco_dict, marker_size, output_folder)