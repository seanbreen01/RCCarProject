import cv2
import numpy as np
import time

gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, -1, (25,25), 5)
image_gpu = cv2.cuda_GpuMat()   #declaring CUDA object into which we can pass images for processing with onboard GPU

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

startTime = time.time()

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
window_title = "RCCar Video input feed"

def imageProcessing(frame):
    image_gpu.upload(frame)  

    gray_gpu = cv2.cuda.cvtColor(image_gpu, cv2.COLOR_BGR2GRAY)

    #Do some more with frame, etc. etc. 



    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Will actually return a command to send to the Arduino and affect some change in its movement
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
                    print('Show stuff bit')

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