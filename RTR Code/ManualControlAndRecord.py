import cv2
from pynput import keyboard
import threading
from keyboard_handler import on_press

def capture_video():
    # Define the GStreamer pipeline for the CSI camera
    gstreamer_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

    # Create a VideoCapture object
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

    #cap = cv2.VideoCapture(0) # if want to use webcam for debug, change 'out' resolution to 640*480 also

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920, 1080))

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Write the frame into the file 'output.avi'
        out.write(frame)

        # Display the resulting frame
        #cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    # Start video capture in a separate thread
    video_thread = threading.Thread(target=capture_video)
    video_thread.start()

    # Start listening for keyboard input
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    video_thread.join()
    listener.join()

if __name__ == "__main__":
    main()