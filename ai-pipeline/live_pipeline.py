import sys
import os
import json
import cv2
import argparse
import math
from collections import defaultdict
from ultralytics import YOLO

# ==========================================
# 1. CLI ARGUMENTS & SETUP
# ==========================================



# --- Configuration Constants ---
MODEL_PATH = "yolov8n.pt"
PROCESSING_FPS = 10
CONFIDENCE_THRESHOLD = 0.35
DEVICE = "0"
TARGET_CLASSES = ["person", "car", "motorcycle", "bus", "truck", "bicycle"]


def log(message):
    """Write diagnostic output to stderr only, keeping stdout clean for JSON output."""
    sys.stderr.write(f"[PYTHON AI] {message}\n")
    sys.stderr.flush()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Real-Time YOLOv8 Smart City Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--upload_id", type=int, required=False, default=0, help="Database Upload ID")
    parser.add_argument("--model", type=str, required=False, default=MODEL_PATH, help="Path to YOLO model weights")
    parser.add_argument("--processing_fps", type=int, required=False, default=PROCESSING_FPS, help="Target FPS for inference")
    parser.add_argument("--confidence", type=float, required=False, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--device", type=str, required=False, default=DEVICE, help="Execution device (e.g., '0' for GPU, 'cpu')")
    parser.add_argument("--target_classes", type=str, nargs="+", required=False, default=TARGET_CLASSES, help="List of target classes to detect")
    return parser.parse_args(argv)


def validate_config(cfg):
    """
    Validates CLI configuration before any heavy initialization occurs.
    Exits the process with a clean stderr message on failure.
    """
    if cfg.processing_fps <= 0:
        log(f"Error: Invalid processing_fps ({cfg.processing_fps}). Must be > 0.")
        sys.exit(1)

    if not (0.0 <= cfg.confidence <= 1.0):
        log(f"Error: Invalid confidence threshold ({cfg.confidence}). Must be between 0.0 and 1.0.")
        sys.exit(1)

    if not cfg.target_classes:
        log("Error: target_classes cannot be empty.")
        sys.exit(1)

    if not os.path.isfile(cfg.video):
        log(f"Error: Video file does not exist at path: {cfg.video}")
        sys.exit(1)

    cap_check = cv2.VideoCapture(cfg.video)
    if not cap_check.isOpened():
        log(f"Error: Could not open/read video file at path: {cfg.video}")
        cap_check.release()
        sys.exit(1)
    cap_check.release()


# --- Parse & Validate ---
args = parse_args()
validate_config(args)

# --- Normalized config for downstream sections ---
# Later sections (YOLO init, video processing, IncidentEngine, JSON output)
# should read from this dict rather than re-parsing args, keeping the
# pipeline consistent and easy to extend/test.
CONFIG = {
    "video_path": args.video,
    "upload_id": args.upload_id,
    "model_path": args.model,
    "processing_fps": args.processing_fps,
    "confidence": args.confidence,
    "device": args.device,
    "target_classes": args.target_classes,
}

log(f"Initializing YOLOv8 AI Pipeline for Upload ID: {CONFIG['upload_id']}")
log(f"Video Source: {CONFIG['video_path']}")
log(f"Config -> Model: {CONFIG['model_path']}, Target FPS: {CONFIG['processing_fps']}, "
    f"Confidence: {CONFIG['confidence']}, Device: {CONFIG['device']}")
log(f"Target Classes: {CONFIG['target_classes']}")

# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================
import torch
from ultralytics import YOLO

try:
    requested_device = args.device

    # --- Resolve actual execution device ---
    if requested_device.lower() == "cpu":
        actual_device = "cpu"
        log("CPU execution explicitly requested. Bypassing CUDA checks.")
    elif torch.cuda.is_available():
        actual_device = requested_device
        log(f"CUDA is available. Hardware acceleration enabled on device: '{actual_device}'")
    else:
        actual_device = "cpu"
        log(
            f"WARNING: GPU (device '{requested_device}') was requested, "
            f"but CUDA is unavailable. Gracefully falling back to CPU."
        )

    # --- Load model ---
    log(f"Attempting to load YOLO model weights from: {args.model}...")
    model = YOLO(args.model)

    # Ultralytics places the model on CPU by default at load time; only move it
    # explicitly when a non-default (GPU) device was actually resolved, avoiding
    # a redundant/no-op device transfer.
    if actual_device != "cpu":
        model.to(actual_device)

    log(f"YOLO model '{args.model}' loaded and deployed to '{actual_device}' successfully.")

    model_metadata = {
        "model_name": args.model,
        "device": actual_device,
        "confidence_threshold": args.confidence,
    }
    log(f"Initialization Complete -> Active Config: {model_metadata}")

except Exception as e:
    log(f"CRITICAL ERROR: Failed to load YOLO model. Details: {e}")
    sys.exit(1)
# ==========================================
# 3. INCIDENT ENGINE (MODULAR RULES)
# ==========================================
class IncidentEngine:
    """
    Modular, rule-based incident detector.

    Each rule (stopped vehicle, crowd detection, ...) is implemented as its
    own small state machine so future rules (accident detection, restricted
    zone entry, emergency vehicle detection, etc.) can be added as additional
    `_analyze_<rule>` methods without touching existing logic.
    """

    # --- Tunable configuration (kept local to the engine, not CLI-exposed) ---
    HISTORY_WINDOW_SECONDS = 3.0     # rolling window kept per track
    STOPPED_SPEED_THRESHOLD = 10.0   # px/sec below which a vehicle is "slow"
    STOPPED_MIN_DURATION = 5.0       # seconds of sustained low speed to confirm "stopped"
    CROWD_MIN_PERSONS = 6            # minimum simultaneous persons to consider a crowd
    CROWD_MIN_DURATION = 5.0         # seconds a crowd must persist before reporting
    STALE_TRACK_TIMEOUT = 10.0       # seconds unseen before a track's state is purged

    def __init__(self):
        # track_id -> list of (cx, cy, second) samples, pruned to HISTORY_WINDOW_SECONDS
        self.track_history = defaultdict(list)

        # track_id -> {"state": ..., "stopped_since": ...} for vehicle motion rule
        self.vehicle_states = {}

        # track_id -> last second this track was observed (for stale cleanup)
        self.last_seen_track = {}

        # dedupe key set so the same physical incident isn't reported twice
        self.reported_incidents = set()

        # crowd rule state (single global state machine, not per-track)
        self.crowd_state = "MONITORING"       # MONITORING | POTENTIAL_CROWD | CROWD_CONFIRMED
        self.crowd_start_second = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def calculate_distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def _prune_history(self, track_id, current_second):
        """Keep only the last HISTORY_WINDOW_SECONDS of samples, based on
        actual timestamps rather than a fixed sample count, so behavior is
        independent of the pipeline's actual processing FPS."""
        history = self.track_history[track_id]
        cutoff = current_second - self.HISTORY_WINDOW_SECONDS
        while history and history[0][2] < cutoff:
            history.pop(0)

    def _cleanup_stale_tracks(self, current_second):
        """Drop history/state for tracks that haven't been seen in a while,
        so memory doesn't grow unbounded over a long video."""
        stale_ids = [
            tid for tid, last_seen in self.last_seen_track.items()
            if current_second - last_seen > self.STALE_TRACK_TIMEOUT
        ]
        for tid in stale_ids:
            self.track_history.pop(tid, None)
            self.vehicle_states.pop(tid, None)
            self.last_seen_track.pop(tid, None)
            # Free up the dedupe key so a *new* track later reusing this id
            # (or a re-detected object) can be reported again if warranted.
            self.reported_incidents.discard(f"STOPPED_VEHICLE_{tid}")

    # ------------------------------------------------------------------
    # Rule: Stopped / stalled vehicle
    # ------------------------------------------------------------------
    def _analyze_vehicle(self, track, track_id, obj_class, frame_num, second, timeline_output):
        history = self.track_history[track_id]
        if len(history) < 2:
            return

        start_pos = history[0]
        curr_pos = history[-1]
        dist_moved = self.calculate_distance(start_pos, curr_pos)
        time_elapsed = curr_pos[2] - start_pos[2]
        if time_elapsed <= 0:
            return

        # This is raw pixel-space speed, NOT a real-world speed (km/h, mph, ...).
        # No camera calibration is available in this pipeline, so we do not
        # attempt to fabricate a real-world velocity.
        speed_px_per_sec = dist_moved / time_elapsed
        track['speed_px_per_sec'] = round(speed_px_per_sec, 2)

        state_info = self.vehicle_states.setdefault(track_id, {
            "state": "MOVING",
            "stopped_since": None,
        })

        if speed_px_per_sec < self.STOPPED_SPEED_THRESHOLD:
            if state_info["state"] == "MOVING":
                state_info["state"] = "POTENTIALLY_STOPPED"
                state_info["stopped_since"] = second

            elif state_info["state"] == "POTENTIALLY_STOPPED":
                duration = second - state_info["stopped_since"]
                if duration >= self.STOPPED_MIN_DURATION:
                    state_info["state"] = "STOPPED_CONFIRMED"
                    incident_key = f"STOPPED_VEHICLE_{track_id}"
                    if incident_key not in self.reported_incidents:
                        self.reported_incidents.add(incident_key)
                        # Derived, rule-based score (not a model confidence value).
                        incident_score = round(
                            min(1.0, duration / (self.STOPPED_MIN_DURATION * 2)), 2
                        )
                        timeline_output.append({
                            "frame": frame_num,
                            "second": second,
                            "type": "TRAFFIC_INCIDENT",
                            "description": f"Stopped or stalled {obj_class} detected in traffic lane.",
                            "payload": {
                                "track_id": track_id,
                                "incident_score": incident_score,
                                "stopped_duration_sec": round(duration, 1),
                                "severity": "Medium",
                            }
                        })
            # STOPPED_CONFIRMED: remain confirmed, do not re-report every frame.

        else:
            # Speed recovered above threshold.
            if state_info["state"] in ("POTENTIALLY_STOPPED", "STOPPED_CONFIRMED"):
                state_info["state"] = "RESUMED"
                # Allow a future stop of this same track to be reported again.
                self.reported_incidents.discard(f"STOPPED_VEHICLE_{track_id}")
            else:
                state_info["state"] = "MOVING"
            state_info["stopped_since"] = None

    # ------------------------------------------------------------------
    # Rule: Crowd detection
    # ------------------------------------------------------------------
    def _analyze_crowd(self, person_count, frame_num, second, timeline_output):
        if person_count >= self.CROWD_MIN_PERSONS:
            if self.crowd_state == "MONITORING":
                self.crowd_state = "POTENTIAL_CROWD"
                self.crowd_start_second = second

            elif self.crowd_state == "POTENTIAL_CROWD":
                duration = second - self.crowd_start_second
                if duration >= self.CROWD_MIN_DURATION:
                    self.crowd_state = "CROWD_CONFIRMED"
                    incident_key = f"CROWD_{int(self.crowd_start_second)}"
                    if incident_key not in self.reported_incidents:
                        self.reported_incidents.add(incident_key)
                        incident_score = round(
                            min(1.0, person_count / (self.CROWD_MIN_PERSONS * 2)), 2
                        )
                        timeline_output.append({
                            "frame": frame_num,
                            "second": second,
                            "type": "CROWD_DETECTED",
                            "description": f"Dense crowd gathering detected ({person_count} persons).",
                            "payload": {
                                "person_count": person_count,
                                "incident_score": incident_score,
                                "duration_sec": round(duration, 1),
                                "severity": "Low",
                            }
                        })
            # CROWD_CONFIRMED: stays confirmed, no repeated event each second.

        else:
            # Crowd dispersed below threshold; reset so a future gathering
            # can be detected and reported again.
            self.crowd_state = "MONITORING"
            self.crowd_start_second = None

    # ------------------------------------------------------------------
    # Entry point (interface preserved for callers)
    # ------------------------------------------------------------------
    def analyze(self, frame_num, second, tracks, detections_output, timeline_output):
        person_count = 0

        for track in tracks:
            track_id = track['track_id']
            obj_class = track['class']
            bbox = track['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            self.last_seen_track[track_id] = second
            self.track_history[track_id].append((cx, cy, second))
            self._prune_history(track_id, second)

            if obj_class == 'person':
                person_count += 1

            if obj_class in ('car', 'bus', 'truck'):
                self._analyze_vehicle(track, track_id, obj_class, frame_num, second, timeline_output)

        self._analyze_crowd(person_count, frame_num, second, timeline_output)
        self._cleanup_stale_tracks(second)

# ==========================================
# 4. VIDEO PROCESSING LOOP
# ==========================================
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    log("Error: Could not open video file.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or math.isnan(fps):
    fps = 30.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Use configured processing FPS from CLI args instead of a hardcoded value.
target_processing_fps = args.processing_fps
frame_skip = max(1, int(round(fps / target_processing_fps)))

incident_engine = IncidentEngine()
final_results = {
    "duration": round(total_frames / fps, 2) if fps > 0 else 0,
    "tracking": [],
    "timeline": []
}
final_results["timeline"].append({
    "frame": 1,
    "second": 0,
    "type": "YOLO_INIT",
    "description": f"YOLOv8 initialized. Analyzing video at {target_processing_fps} FPS.",
    "payload": model_metadata
})

frame_count = 0
processed_count = 0
LOG_INTERVAL_PROCESSED_FRAMES = 50  # avoid spamming stderr every single frame

log(
    f"Starting inference... FPS: {fps}, "
    f"Total Frames: {total_frames}, "
    f"Frame Skip: {frame_skip}, "
    f"Target Classes: {args.target_classes}"
)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        processed_count += 1
        current_second = round(frame_count / fps, 2)

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.confidence,
            device=actual_device,
            verbose=False
        )

        current_tracks = []
        if results and len(results) > 0 and results[0].boxes:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls.item())
                conf = float(box.conf.item())
                obj_class = model.names[class_id]

                if obj_class not in args.target_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detection_data = {
                    "frame": frame_count,
                    "second": current_second,
                    "class": obj_class,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                }

                # Only promote a detection to a tracking record when ByteTrack
                # actually assigned a valid persistent track ID. Never invent one.
                if box.id is not None:
                    track_data = dict(detection_data)
                    track_data["track_id"] = int(box.id.item())
                    track_data["speed_px_per_sec"] = 0
                    current_tracks.append(track_data)
                    final_results["tracking"].append(track_data)

        if current_tracks:
            incident_engine.analyze(
                frame_count,
                current_second,
                current_tracks,
                final_results["tracking"],
                final_results["timeline"]
            )

        if processed_count % LOG_INTERVAL_PROCESSED_FRAMES == 0:
            log(f"Progress: processed {processed_count} frames (video frame {frame_count}/{total_frames}, t={current_second}s)")

finally:
    cap.release()

final_results["timeline"].append({
    "frame": frame_count,
    "second": final_results["duration"],
    "type": "PROCESSING_COMPLETE",
    "description": f"AI extraction complete. Processed {processed_count} frames.",
    "payload": {
        "total_detections": len(final_results["tracking"])
    }
})
log("Inference complete. Exporting JSON to Node.js backend.")
# ==========================================
# 5. STDOUT EXPORT (MUST BE STRICT JSON)
# ==========================================
# Node.js relies on this exact print statement to parse the database payload.
# CRITICAL: stdout must contain ONLY the final JSON. All diagnostics stay on
# stderr via log(). Nothing else may be printed here.

try:
    tracking_records = final_results.get("tracking", [])
    timeline_records = final_results.get("timeline", [])

    unique_track_ids = {
        t["track_id"] for t in tracking_records if "track_id" in t
    }

    object_counts = defaultdict(int)
    for t in tracking_records:
        obj_class = t.get("class")
        if obj_class:
            object_counts[obj_class] += 1

    # Incidents are timeline entries that aren't purely informational markers.
    NON_INCIDENT_TYPES = {"YOLO_INIT", "PROCESSING_COMPLETE"}
    incident_count = sum(
        1 for e in timeline_records if e.get("type") not in NON_INCIDENT_TYPES
    )

    summary = {
        "total_detections": len(tracking_records),
        "unique_tracks": len(unique_track_ids),
        "incident_count": incident_count,
        "object_counts": dict(object_counts),
    }

    metadata = {
        "model": model_metadata.get("model_name"),
        "device": model_metadata.get("device"),
        "confidence_threshold": model_metadata.get("confidence_threshold"),
        "input_fps": round(fps, 2),
        "processing_fps": args.processing_fps,
        "total_frames": total_frames,
        "frames_processed": processed_count,
        "duration_sec": final_results.get("duration", 0),
    }

    output = {
        "status": "completed",
        "upload_id": args.upload_id,
        "metadata": metadata,
        "summary": summary,
        "tracking": tracking_records,
        "timeline": timeline_records,
    }

    print(json.dumps(output))
    sys.exit(0)

except Exception as e:
    # Even on failure, stdout must remain strict, parseable JSON so the
    # Node.js backend's parser doesn't choke. Full detail goes to stderr.
    log(f"CRITICAL ERROR: Failed to assemble/export final JSON output. Details: {e}")
    error_output = {
        "status": "error",
        "upload_id": getattr(args, "upload_id", 0),
        "error": str(e),
    }
    print(json.dumps(error_output))
    sys.exit(1)