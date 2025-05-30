# Import necessary modules
from ultralytics import YOLO

# Train the YOLOv8 model
model = YOLO("yolov8n.pt")

results = model.train(
    data="your_dataset_path/coco128.yaml",  # Path to dataset YAML file
    epochs=100,                             # Number of training epochs
    imgsz=640,                              # Input image size
    batch=16,                               # Batch size
    workers=4,                              # Number of data loading workers
    name="yolo8_bike_detect",               # Name for the training run
    device=0                                # Use GPU 0; set to 'cpu' for CPU training
)

# Path to the trained model weights
trained_model_path = results.save_dir / "weights/bike.pt"
print(f"Trained model saved at: {trained_model_path}")
