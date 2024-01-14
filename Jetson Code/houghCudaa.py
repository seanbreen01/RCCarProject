import cv2
import numpy as np
import matplotlib as plt

# Load video
cap = cv2.VideoCapture('./20231023_172356.mp4')

# Check if video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create CUDA Hough Line detector
cuda_detector = cv2.cuda.createHoughSegmentDetector(rho=1, theta=np.pi / 180, minVotes=100)

# Initialize VideoWriter (optional)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Upload frame to GPU
    d_frame = cv2.cuda_GpuMat()
    d_frame.upload(frame)

    # Convert to grayscale
    d_gray = cv2.cuda.cvtColor(d_frame, cv2.COLOR_BGR2GRAY)

    # Detect lines
    lines = cuda_detector.detect(d_gray)

    # Download lines from GPU
    lines = lines.download()

    # Draw lines on frame
    for line in lines[0]:
        rho, theta = line[0], line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

    # Write frame to output video
    out.write(frame)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()