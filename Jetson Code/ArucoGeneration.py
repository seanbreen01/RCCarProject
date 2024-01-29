import cv2
import cv2.aruco as aruco


def generate_aruco_markers(dictionary, marker_size, output_folder):
    for marker_id in range(len(dictionary.bytesList)):
        marker_image = aruco.drawMarker(dictionary, marker_id, marker_size)
        cv2.imwrite(f"{output_folder}/marker_{marker_id}.png", marker_image)

if __name__ == "__main__":
    aruco_dict = aruco.Dictionary(aruco.DICT_4X4_50)
    marker_size = 200  # size of the marker in pixels
    output_folder = "Aruco Markers"  # specify your output folder here
    generate_aruco_markers(aruco_dict, marker_size, output_folder)