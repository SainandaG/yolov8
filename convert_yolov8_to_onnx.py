from ultralytics import YOLO

# Load the YOLOv8 model (example: yolov8m.pt)
model = YOLO("yolov8m.pt")  # Replace with the correct path to your .pt model

# Export the model to ONNX format with the specified input size
model.export(format="onnx", imgsz=[480, 640])  # Adjust the image size if necessary
