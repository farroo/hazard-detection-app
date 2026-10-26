import cv2
import numpy as np
from ultralytics import YOLO

class HazardDetector:
    def __init__(self, model_path='best.pt'):
        # Load your custom fine-tuned regional model weights
        self.model = YOLO(model_path)
        # Store previous frame centroids for wrong-way tracking: {track_id: (x, y)}
        self.prev_centroids = {}
        
    def process_frame(self, frame):
        # Enforce Spatial Normalization (Resize input frame to 640x640 for fast inference)
        frame = cv2.resize(frame, (640, 640))
        
        height, width, _ = frame.shape
        
        # Run YOLO tracking (ByteTrack) across frames
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # Draw a visual safety threshold line (bottom 30% of the frame)
        hazard_line_y = int(height * 0.70)
        cv2.line(frame, (0, hazard_line_y), (width, hazard_line_y), (0, 0, 255), 2)
        cv2.putText(frame, "CRITICAL PROXIMITY ZONE", (20, hazard_line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        current_centroids = {}
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = self.model.names[cls_id]
                
                # Calculate current centroid (center of bounding box base)
                cx = int((x1 + x2) / 2)
                cy = int(y2)
                current_centroids[track_id] = (cx, cy)
                
                # 1. Proximity Hazard Check (Evaluates bounding box coordinates relative to the bottom threshold)
                is_close_proximity = y2 > hazard_line_y
                
                # 2. Wrong-Way Movement Check (Cross-verifies actor movement vectors against expected traffic direction)
                is_wrong_way = False
                if track_id in self.prev_centroids:
                    prev_cx, prev_cy = self.prev_centroids[track_id]
                    if (cy - prev_cy) > 15:  # Moving rapidly downwards towards the camera
                        is_wrong_way = True
                
                # Determine Box Color & Alert Message
                if is_wrong_way:
                    color = (0, 0, 255) # Red for Wrong-Way Hazard
                    alert_text = f"ID:{track_id} WRONG WAY!"
                elif is_close_proximity:
                    color = (0, 165, 255) # Orange for High Proximity Warning
                    alert_text = f"ID:{track_id} CLOSE PROXIMITY"
                else:
                    color = (0, 255, 0) # Green for Safe
                    alert_text = f"ID:{track_id} {label}"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, alert_text, (x1, max(y1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # Update centroid history
        self.prev_centroids = current_centroids
        return frame