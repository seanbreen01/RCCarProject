import sys
import select
import tty
import termios
import cv2
import smbus
import time

# Global flag to control video recording
stop_video = False

# Nvidia Jetson Nano i2c Bus 0
bus = smbus.SMBus(0)
# This is the address we setup in the Arduino script
address = 0x40

def writeToArduino(valueToWrite):
    bytesToWrite = []
    for value in valueToWrite:
        # Split each integer into two bytes (high byte and low byte)
        highByte = (value >> 8) & 0xFF
        lowByte = value & 0xFF
        bytesToWrite.extend([highByte, lowByte])

    # Send the byte array
    print('Bytes: ')    #For debug
    print(bytesToWrite)
    bus.write_i2c_block_data(address, 0, bytesToWrite)

def readNumber():
    number = bus.read_byte(address)
    # number = bus.read_byte_data(address, 1)

    #TODO update to read from bus in correct format
    return number

def is_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def main():
    global stop_video

    xres = 1920
    yres = 1080
    frames = 30

    # Define the GStreamer pipeline for the CSI camera
    gstreamer_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int){xres}, height=(int){yres}, format=(string)NV12, framerate=(fraction){frames}/1 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, width=(int){xres}, height=(int){yres}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

    cap = cv2.VideoCapture(gstreamer_pipeline.format(xres=xres, yres=yres, frames=frames), cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (xres, yres))

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        window_handle = cv2.namedWindow("debug", cv2.WINDOW_AUTOSIZE)
        print("Press 'W', 'A', 'S', 'D' keys (Enter to exit):")

        while not stop_video:
            ret, frame = cap.read()
            if cv2.getWindowProperty("debug", cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow("debug", frame)
            else:
                break 
            if not ret:
                print("Can't receive frame. Exiting ...")
                break
            #out.write(frame)

            if is_data():
                c = sys.stdin.read(1)
                if c == '\n':  # Enter key to break the loop
                    break
                if c.upper() == 'W':
                    print("Forward")    
                    writeToArduino([1,1600,250])
                elif c.upper() == 'A':
                    print("Left")
                    writeToArduino([0,110,250])
                elif c.upper() == 'S':
                    print("Backward")
                    writeToArduino([1,1400,250])
                elif c.upper() == 'D':
                    print("Right")
                    writeToArduino([0,60,250])

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        stop_video = True
        cap.release()
        out.release()
        print("Exited gracefully")

if __name__ == "__main__":
    filename = input('Input filename for video recording: ')
    filename = str(filename) + '.avi'

    # Resolution selection code here (unchanged)
    # Framerate selection code here (unchanged)

    main()
