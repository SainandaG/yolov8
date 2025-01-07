import cv2
import requests
import numpy as np
from io import BytesIO

from yolov8 import YOLOv8

# Function to load image from URL
def imread_from_url(url):
    response = requests.get(url)
    img_arr = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    return img

# Initialize YOLOv8 object detector
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Read image from URL
img_url = "https://i.ytimg.com/vi/1X1WKb6Vgok/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLDVSqKj7cxIKx_g9EJdS8DLOUdxXw"  # Replace with a valid direct image URL
img = imread_from_url(img_url)

# Check if the image is loaded correctly
if img is None:
    print("Failed to load image from URL.")
else:
    # Detect Objects
    boxes, scores, class_ids = yolov8_detector(img)

    # Draw detections
    combined_img = yolov8_detector.draw_detections(img)
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Objects", combined_img)
    cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
    cv2.waitKey(0)
