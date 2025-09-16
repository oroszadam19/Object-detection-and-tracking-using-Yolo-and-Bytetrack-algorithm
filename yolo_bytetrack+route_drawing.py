import supervision as sv
import numpy as np
from ultralytics import YOLO
import cv2
import random

model = YOLO('yolov8n.pt')
byte_tracker = sv.ByteTrack()

def generate_colors(num_classes):
    random.seed(42)
    colors = {}
    for class_id in range(num_classes):
        colors[class_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return colors

class_colors = generate_colors(len(model.model.names))

trajectories = {}

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    tracked_detections = byte_tracker.update_with_detections(detections)

    annotated_frame = frame.copy()

    for i in range(len(tracked_detections.xyxy)):
        x1, y1, x2, y2 = map(int, tracked_detections.xyxy[i])
        tracker_id = tracked_detections.tracker_id[i]
        class_id = int(tracked_detections.class_id[i])
        confidence = tracked_detections.confidence[i]

        label = f"#{tracker_id} {model.model.names[class_id]} {confidence:.2f}"
        color = class_colors[class_id]

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if tracker_id not in trajectories:
            trajectories[tracker_id] = []
        trajectories[tracker_id].append((center_x, center_y))

        for j in range(1, len(trajectories[tracker_id])):
            cv2.line(
                annotated_frame,
                trajectories[tracker_id][j - 1],
                trajectories[tracker_id][j],
                color,
                2
            )

    return annotated_frame


sv.process_video(
    source_path='C:\\Users\\orosz\\Videos\\traffic.mp4',
    target_path="result_new.mp4",
    callback=callback
)
