import cv2
import numpy as np
from ultralytics import YOLO

class HazardDetector:
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)
        self.prev_centroids = {}
        self.prev_box_heights = {}
        
        # Session Analytics Counters
        self.analytics = {
            "total_frames": 0,
            "wrong_way_count": 0,
            "proximity_count": 0,
            "ttc_critical_count": 0,
            "detected_classes": {}
        }
        
    def process_frame(self, frame, zone_width=0.15, zone_horizon=0.55):
        frame = cv2.resize(frame, (640, 640))
        height, width, _ = frame.shape
        self.analytics["total_frames"] += 1
        
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # Run YOLO tracking with ByteTrack
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # 2. Multi-Zone Polygon Risk Mapping (Trapezoid zone targeting oncoming road space)
        # Dynamic polygon zone based on UI sliders
        zone_polygon = np.array([
            [int(width * zone_width), int(height * zone_horizon)],             # Top-left
            [int(width * (1.0 - zone_width)), int(height * zone_horizon)],     # Top-right
            [int(width * 0.95), height],                                        # Bottom-right
            [int(width * 0.05), height]                                         # Bottom-left
        ], dtype=np.int32)

        # Draw Polygon Zone on frame
        cv2.polylines(frame, [zone_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(frame, "MONITORED HAZARD ZONE", (zone_polygon[0][0], zone_polygon[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        current_centroids = {}
        current_box_heights = {}
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = self.model.names[cls_id]
                
                # Track class frequencies for analytics report
                self.analytics["detected_classes"][label] = self.analytics["detected_classes"].get(label, 0) + 1
                
                cx = int((x1 + x2) / 2)
                cy = int(y2)
                box_h = y2 - y1
                
                current_centroids[track_id] = (cx, cy)
                current_box_heights[track_id] = box_h
                
                # Check Polygon Zone containment
                in_polygon_zone = cv2.pointPolygonTest(zone_polygon, (cx, cy), False) >= 0
                
                # 3. Wrong-Way Movement Check
                is_wrong_way = False
                if track_id in self.prev_centroids:
                    prev_cx, prev_cy = self.prev_centroids[track_id]
                    if (cy - prev_cy) > 12:  # Moving rapidly downwards toward camera
                        is_wrong_way = True
                        self.analytics["wrong_way_count"] += 1
                        
                # 4. Time-to-Collision (TTC) & Velocity Estimation
                ttc_seconds = 99.0
                if track_id in self.prev_box_heights:
                    prev_h = self.prev_box_heights[track_id]
                    height_growth_rate = box_h - prev_h  # Positive value means approaching
                    if height_growth_rate > 0:
                        remaining_distance_proxy = height - y2
                        closing_speed = height_growth_rate
                        ttc_seconds = max(0.1, remaining_distance_proxy / (closing_speed * 30.0))
                
                is_critical_ttc = ttc_seconds < 1.5  # Critical collision warning threshold (< 1.5s)
                if is_critical_ttc and in_polygon_zone:
                    self.analytics["ttc_critical_count"] += 1

                if in_polygon_zone:
                    self.analytics["proximity_count"] += 1

                # Determine Alert Styling
                if is_wrong_way:
                    color = (0, 0, 255) # Red
                    alert_text = f"ID:{track_id} WRONG WAY!"
                elif is_critical_ttc and in_polygon_zone:
                    color = (0, 0, 255) # Red
                    alert_text = f"ID:{track_id} TTC:{ttc_seconds:.1f}s CRITICAL!"
                elif in_polygon_zone:
                    color = (0, 165, 255) # Orange
                    alert_text = f"ID:{track_id} IN ZONE ({label})"
                else:
                    color = (0, 255, 0) # Green
                    alert_text = f"ID:{track_id} {label}"
                
                # Render annotations
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, alert_text, (x1, max(y1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                
        self.prev_centroids = current_centroids
        self.prev_box_heights = current_box_heights
        return frame

    def get_incident_summary(self):
        """Returns structured session analytics for the dashboard report."""
        return self.analytics