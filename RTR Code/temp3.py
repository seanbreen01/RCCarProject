import cv2

# def gstreamer_pipeline(
#     sensor_id=0,
#     capture_width=1920,
#     capture_height=1080,
#     display_width=960,
#     display_height=540,
#     framerate=30,
#     flip_method=0,
# ):
#     return (
#         "nvarguscamerasrc sensor-id=%d ! "
#         "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
#         "nvvidconv flip-method=%d ! "
#         "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#         "videoconvert ! "
#         "video/x-raw, format=(string)BGR ! appsink"
#         % (
#             sensor_id,
#             capture_width,
#             capture_height,
#             framerate,
#             flip_method,
#             display_width,
#             display_height,
#         )
#     )

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

    filename = input('Input filename for video recording: ')
    filename = str(filename) + '.avi'

    resolution = input('Input resolution: 1 for 1920x1080, 2 for 1280x720, 3 for 640x480\n')
    if resolution == '1':
        print('Resolution set to 1920x1080')
        xres = 1920
        yres = 1080
    elif resolution == '2':
        print('Resolution set to 1280x720')
        xres = 1280
        yres = 720
    elif resolution == '3':
        print('Resolution set to 640x480')
        xres = 640
        yres = 480
    else:
        print('Invalid input, setting resolution to 1920x1080')
        resolution = '1'
        xres = 1920
        yres = 1080

    framerate = input('Input framerate: 1 for 30fps, 2 for 60fps\n')
    if framerate == '1':
        print('Framerate set to 30fps')
        frames = 30	
    elif framerate == '2':
        print('Framerate set to 60fps')
        frames = 60
    else:
        print('Invalid input, setting framerate to 30fps')
        framerate = '1'
        frames = 30

    pipeline = gstreamer_pipeline(capture_width=xres, capture_height=yres, display_width=xres, display_height=yres, framerate=frames, flip_method=2)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print('no open')
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, float(frames)-20, (xres,yres))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('no ret')
                break

            out.write(frame)

            #cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()