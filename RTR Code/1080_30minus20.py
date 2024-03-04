import cv2
import sys
import select
import tty
import termios
import smbus
import time


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

def gstreamer_pipeline(sensor_id=0, sensor_mode=3, capture_width=640, capture_height=480, display_width=640, display_height=480, framerate=30, flip_method=2):
    return (
        f'nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! '
        f'video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, '
        f'format=(string)NV12, framerate=(fraction){framerate}/1 ! '
        f'nvvidconv flip-method={flip_method} ! '
        f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! '
        f'videoconvert ! '
        f'video/x-raw, format=(string)BGR ! appsink'
    )

def main():

    
    filename = '1080_30minus20.avi'

    xres = 1920
    yres = 1080
    frames = 30

    pipeline = gstreamer_pipeline(capture_width=xres, capture_height=yres, display_width=xres, display_height=yres, framerate=frames, flip_method=2)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    old_settings = termios.tcgetattr(sys.stdin)

    if not cap.isOpened():
        print('no open')
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, float(frames)-20, (xres,yres))

    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            ret, frame = cap.read()
            if not ret:
                print('no ret')
                break

            out.write(frame)

            if is_data():
                c = sys.stdin.read(1)
                if c == '\n':  # Enter key to break the loop
                    break
                if c.upper() == 'W':
                    print("Forward")    
                    writeToArduino([1,1600,500])
                    #Could add sliding increase to 1600 value to gradually accelerate, prob not necessary for needs here though 
                elif c.upper() == 'A':
                    print("Left")
                    writeToArduino([0,110,500])
                elif c.upper() == 'S':
                    print("Backward")
                    writeToArduino([1,1400,500])
                elif c.upper() == 'D':
                    print("Right")
                    writeToArduino([0,60,500])

            #cv2.imshow('frame', frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()