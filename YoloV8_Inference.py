# https://www.youtube.com/watch?v=o4Zd-IeMlSY

import cv2
from ultralytics import YOLO
import random

# load the YOLOv8 model
#model = YOLO("yolov8n-seg.pt")  # load a pretrained model
model = YOLO("yolov8n.pt")  # load a pretrained model
#model = YOLO("yolov8s-seg.pt")  # load a pretrained model
#model = YOLO("yolov8m-seg.pt")  # load a pretrained model
#model = YOLO("yolov8l-seg.pt")  # load a pretrained model

# or load a Own trained model
#model = YOLO("D:\\Pirooz\\0 handwrite detector\\runs\\detect\\train56\\weights\\best.pt")  # load a pretrained model


# Open the video file
# video_path ="d:/Driving in Central Prague Czechia - the Heart of Europe - 4K City Drive.mp4"
video_path ="D:\\Pirooz testing\\20240222_103752.mp4" # wideq
#video_path ="D:\\Pirooz testing\\20240222_104259.mp4" # normal



cap=cv2.VideoCapture(video_path)
#cap=cv2.VideoCapture(0)


# Get the original width and height of the video
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the new width and height while preserving aspect ratio
target_width = 640
target_height = int(original_height * target_width / original_width)

# Set the initial frame number to a random value
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#start_frame = random.randint(0, total_frames - 1)

#Jump to frame you want of vide
#cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#cap.set(cv2.CAP_PROP_POS_FRAMES, 4000)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Resize the frame
        frame = cv2.resize(frame, (target_width, target_height))

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        # Display the annotated frame
        cv2.imshow("Annotated Frame", annotated_frame)

        # Display the original frame
        cv2.imshow("Original Frame", frame)


        # Resize the window
        #cv2.resizeWindow("iPilot v1", 640,480)


        # Break the loop if 'q' is pressed 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 

    else:
        # Beak the loop if the end of the video is reached 
        break

# Releas the video capture object and close the display windows 
cap.release()
cv2.destroyAllWindows()


