from ultralytics import YOLO
import os
import cv2
import pandas as pd
from datetime import datetime

# Setup
rider_paths = []
plate_paths = []
saved_rider = 0

# Create timestamped folder
date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_folder = os.path.join("track", date)
os.makedirs(output_folder, exist_ok=True)

model = YOLO("/best.pt")  # Path to the trained YOLO model
video_path = "/5207049-uhd_2160_3840_25fps.mp4"  # Path to the input video for object detection
cap = cv2.VideoCapture(video_path)  # Open the video file for processing


# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer
output_video_path = os.path.join(output_folder, "output_detected_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened() and saved_rider < 5:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)[0]
    boxes = results.boxes

    if boxes is None:
        continue

    all_detections = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_name = model.names[cls_id]
        all_detections.append((class_name, x1, y1, x2, y2))

        # Draw on video
        color = (0, 255, 0) if class_name != "without helmet" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Check for "without helmet" and find corresponding rider and plate
    for det in all_detections:
        if det[0] == "without helmet":
            helmetless_box = det[1:]

            # Find overlapping/nearby rider
            rider_box = None
            for other in all_detections:
                if other[0] == "rider":
                    rx1, ry1, rx2, ry2 = other[1:]
                    # Check if center of helmetless is inside rider box
                    cx = (helmetless_box[0] + helmetless_box[2]) // 2
                    cy = (helmetless_box[1] + helmetless_box[3]) // 2
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        rider_box = (rx1, ry1, rx2, ry2)
                        break

            # Find number plate in the same frame (not necessarily linked)
            plate_box = None
            for other in all_detections:
                if other[0] == "number plate":
                    plate_box = other[1:]
                    break

            # Save both images if found
            if rider_box and plate_box:
                # Save rider
                rx1, ry1, rx2, ry2 = rider_box
                rider_crop = frame[ry1:ry2, rx1:rx2]
                rider_path = os.path.join(output_folder, f"rider_{saved_rider+1}.jpg")
                cv2.imwrite(rider_path, rider_crop)
                rider_paths.append(rider_path)

                # Save plate
                px1, py1, px2, py2 = plate_box
                plate_crop = frame[py1:py2, px1:px2]
                plate_path = os.path.join(output_folder, f"plate_{saved_rider+1}.jpg")
                cv2.imwrite(plate_path, plate_crop)
                plate_paths.append(plate_path)

                saved_rider += 1
                break  # Move to next frame after one valid pair

    out_video.write(frame)

cap.release()
out_video.release()

# Save CSV
df = pd.DataFrame({
    "rider_paths": rider_paths,
    "number_plate_paths": plate_paths
})
df.to_csv(os.path.join("track", "records.csv"), index=False)

