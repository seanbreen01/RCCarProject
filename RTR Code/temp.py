import sys
import select
import tty
import termios
import threading
import cv2

# Global flag to control video recording thread
stop_video = False

def is_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def record_video(output_file):
    global stop_video
    # Define the GStreamer pipeline for the CSI camera
    gstreamer_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (1920, 1080))

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while not stop_video:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        out.write(frame)

    cap.release()
    out.release()


def main():
    global stop_video
    video_thread = threading.Thread(target=record_video, args=('outputnew23.avi',))
    video_thread.start()

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        print("Press 'W', 'A', 'S', 'D' keys (Enter to exit):")

        while True:
            if is_data():
                c = sys.stdin.read(1)
                if c == '\n':  # Enter key to break the loop
                    break
                if c.upper() == 'W':
                    print("yes")    #Replace with control code eventually
                elif c.upper() == 'A':
                    print("no")
                elif c.upper() == 'S':
                    print("maybe")
                elif c.upper() == 'D':
                    print("thanks")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        stop_video = True
        video_thread.join()
        print("Exited gracefully")

if __name__ == "__main__":
    main()
