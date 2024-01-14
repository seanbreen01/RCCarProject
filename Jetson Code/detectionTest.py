import cv2
import numpy as np
import matplotlib as plt

video = cv2.VideoCapture('./20231023_172356.mp4')

frame_width = 1280  #might be arseways
frame_height = 720

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 30, (frame_width, frame_height))
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break   #escape if no frame from video to look at 
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(grayFrame, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    if lines is not None: 
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 2)

            #cv2.imshow('Frame', frame)     ##display detections on frame
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

video.release()
out.release()
cv2.destroyAllWindows() 