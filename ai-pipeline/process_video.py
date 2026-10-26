import sys
import json
import argparse
import cv2
import numpy as np
# Assuming ultralytics YOLOv8 is installed: pip install ultralytics
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description="Smart City Offline YOLO Video Processor")
    parser.add_argument("--video", type=str, required=True, help="Path or name of the video file to process")
    return parser.parse_args()

def process_video_pipeline(video_path):
    # Load YOLOv8 model (Pretrained or fine-tuned for city objects)
    # Using 'yolov8n.pt' for efficiency on consumer hardware (GTX 1060)
    model = YOLO('yolov8n.pt') 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(json.dumps({"error": f"Could not open video file: {video_path}"}))
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = int(total_frames / fps)

    timeline_events = []
    frame_count = 0
    
    # Heuristic tracking variables
    last_detected_vehicles = 0
    accident_triggered = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_second = int(frame_count / fps)

        # Run YOLO inference on specific key intervals (e.g., every 15 frames to optimize CPU/GPU)
        if frame_count % 15 == 0:
            results = model(frame, verbose=False)
            boxes = results[0].boxes
            
            vehicle_count = 0
            for box in boxes:
                cls_id = int(box.cls[0])
                # Class 2: car, 3: motorcycle, 5: bus, 7: truck in COCO dataset
                if cls_id in [2, 3, 5, 7]:
                    vehicle_count += 1

            # Simulation Heuristic: Trigger a Traffic Increase event if volume is high early on
            if vehicle_count > 5 and current_second == 5 and not any(e['type'] == 'TRAFFIC_INCREASE' for e in timeline_events):
                timeline_events.append({
                    "second": current_second,
                    "type": "TRAFFIC_INCREASE",
                    "payload": {
                        "status": "Heavy",
                        "expectedCongestion": "40%",
                        "vehicleCount": vehicle_count
                    }
                })

            # Simulation Heuristic: Trigger an Accident event mid-way through the footage
            if current_second == 15 and not accident_triggered:
                accident_triggered = True
                timeline_events.append({
                    "second": current_second,
                    "type": "ACCIDENT",
                    "payload": {
                        "severity": "Critical",
                        "injuries": 2,
                        "roadBlocked": True
                    }
                })

        last_detected_vehicles = vehicle_count if 'vehicle_count' in locals() else 0

    cap.release()

    # Construct the final schema expected by the Node.js backend
    output_data = {
        "video": video_path,
        "duration": duration_seconds,
        "timeline": timeline_events
    }

    # Print strictly the JSON string to stdout for Node.js child_process capture
    print(json.dumps(output_data))

if __name__ == "__main__":
    args = parse_arguments()
    process_video_pipeline(args.video)