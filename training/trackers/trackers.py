from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import sys

# Tambahkan jalur ke sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_dir)

# Cetak sys.path untuk memverifikasi
print(sys.path)

try:
    from utils.bbox_utils import get_center_of_bbox, get_bbox_width
except ImportError as e:
    print(f"Error importing bbox_utils: {e}")
    sys.exit(1)

class Tracker:
    def __init__(self, model_path):
        print("Initializing Tracker...")
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            sys.exit(1)
        try:
            self.model = YOLO(model_path)
            self.tracker = sv.ByteTrack()
        except Exception as e:
            print(f"Error initializing model or tracker: {e}")
            sys.exit(1)
        
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            print(f"Processing batch {i//batch_size + 1}")
            try:
                detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
                detections += detections_batch
            except Exception as e:
                print(f"Error during detection: {e}")
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            try:
                with open(stub_path, 'rb') as f:
                    tracks = pickle.load(f)
                return tracks
            except Exception as e:
                print(f"Error reading from stub: {e}")
                sys.exit(1)
        
        detections = self.detect_frames(frames)
        
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
        for frame_num, detection in enumerate(detections):
            print(f"Processing frame {frame_num}")
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            print(f"Class names: {cls_names}")
            
            # Convert to supervision detection format
            try:
                detection_supervision = sv.Detections.from_ultralytics(detection)
            except Exception as e:
                print(f"Error converting to supervision format: {e}")
                continue
            
            # Convert goalkeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            
            # Track objects
            try:
                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            except Exception as e:
                print(f"Error updating tracker: {e}")
                continue
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
            
        if stub_path is not None:
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)
            except Exception as e:
                print(f"Error saving to stub: {e}")
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        return frame
        
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            
            # Draw Players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)
            
            output_video_frames.append(frame)
            
        return output_video_frames

if __name__ == "__main__":
    # Example usage
    model_path = "path_to_your_model.pt"
    tracker = Tracker(model_path)
    # Add your code to load video frames and call tracker methods
