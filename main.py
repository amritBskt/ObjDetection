import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from torchreid.utils import FeatureExtractor
import time
import logging
from scipy.spatial.distance import cosine
from multiprocessing import Process
import sqlite3
from multiprocessing import Manager


# Setup logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG)

# Load YOLOv8 model
model = YOLO("yolov9s.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, nn_budget=100)

# Load ReID model (OSNet)
extractor = FeatureExtractor(
    model_name='osnet_ain_x1_0',
    model_path='osnet_ain_x1_0.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print("[INFO] ReID Model Loaded Successfully!")

# Database setup
def init_db():
    conn = sqlite3.connect("reid_db.sqlite3")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY,
            feature BLOB,
            last_seen_camera INTEGER,
            last_seen_time REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Store rectangles for click detection
rectangles = {}
selected_person = None

def click_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for global_id, rect in rectangles.items():
            x1, y1, x2, y2, color = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if global_id in param:
                    param.remove(global_id)  # Remove from shared list
                else:
                    param.append(global_id)  # Add to shared list
                logging.debug(f"Clicked ID {global_id}, Updated shared list: {param}")


def extract_deep_features(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    person_crop = image[y1:y2, x1:x2]
    if person_crop.size == 0:
        return None  # Avoid processing empty crops
    
    person_crop = cv2.resize(person_crop, (128, 256))
    person_crop = np.transpose(person_crop, (2, 0, 1)) / 255.0
    person_crop = torch.tensor(person_crop, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():  # Ensure we are not computing gradients
        features = extractor(person_crop)
    
    return features[0].cpu().detach().numpy()


def find_matching_person(new_feature, cam_id, threshold=0.7):  # Lower threshold for stricter matching
    if new_feature is None:
        return None

    new_feature = new_feature / np.linalg.norm(new_feature)  # Normalize feature vector
    
    conn = sqlite3.connect("reid_db.sqlite3")
    cursor = conn.cursor()
    cursor.execute("SELECT id, feature FROM persons")
    persons = cursor.fetchall()
    conn.close()

    best_match = None
    best_score = float("inf")
    
    for track_id, old_feature in persons:
        old_feature = np.frombuffer(old_feature, dtype=np.float32)
        old_feature = old_feature / np.linalg.norm(old_feature)  # Normalize stored feature
        
        score = cosine(new_feature, old_feature)
        if score < threshold and score < best_score:
            best_score = score
            best_match = track_id
    
    return best_match

def update_database(local_id, new_feature, cam_id):
    conn = sqlite3.connect("reid_db.sqlite3")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO persons (id, feature, last_seen_camera, last_seen_time)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
        feature = excluded.feature,
        last_seen_camera = excluded.last_seen_camera,
        last_seen_time = excluded.last_seen_time
    """, (local_id, new_feature.tobytes(), cam_id, time.time()))

    conn.commit()
    conn.close()

def process_video(src, cam_id, shared_selected_ids):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Couldn't open video {src}")
        return
    
    print(f"[INFO] Processing video: {src}")
    window_name = f"Camera {cam_id}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_callback, param=shared_selected_ids)
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Use 30 FPS if reading fails

    output_filename = f"output_cam_{cam_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"[INFO] End of video: {src}")
            break  
        
        results = model(frame)
        detections = []
        track_map = {}
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cls == 0 and conf > 0.6:
                    bbox = [x1, y1, x2-x1, y2-y1]
                    deep_features = extract_deep_features(frame, (x1, y1, x2, y2))
                    if deep_features is not None:
                        matched_id = find_matching_person(deep_features, cam_id)
                        global_id = update_database(matched_id, deep_features, cam_id)
                    matched_id = find_matching_person(deep_features, cam_id)
                    # print(matched_id)
                    if matched_id is None:
                        global_id = len(detections) + 1
                    else:
                        global_id = matched_id
                    
                    update_database(global_id, deep_features, cam_id)
                    track_map[len(detections)] = global_id
                    detections.append((bbox, conf, "person"))
        
        tracks = tracker.update_tracks(detections, frame=frame)
        
        for i, track in enumerate(tracks):
            if track.is_confirmed():
                bbox = track.to_tlbr()
                local_id = track.track_id
                # print(local_id)
                global_id = track_map.get(local_id, local_id)

                color = (0, 0, 255) if global_id in shared_selected_ids else (0, 255, 0)

                # color = rectangles.get(global_id, (0, 0, 0, 0, (0, 255, 0)))[4]

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, f"ID {global_id} (Cam {cam_id})", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                rectangles[global_id] = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), color)
        
        cv2.imshow(window_name, frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyWindow(window_name)
    
    if selected_person:
        conn = sqlite3.connect("reid_db.sqlite3")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM persons WHERE id = ?", (selected_person,))
        person_data = cursor.fetchone()
        conn.close()
        print(f"[INFO] Selected Person Data: {person_data}")


def display_tracking_data():
    conn = sqlite3.connect("reid_db.sqlite3")
    cursor = conn.cursor()
    
    # Select only the last_seen_camera and last_seen_time columns
    cursor.execute("SELECT id, last_seen_camera, last_seen_time FROM persons")
    tracked_data = cursor.fetchall()

    conn.close()

    print("\n===== Final Tracked Persons =====")
    print("{:<10} {:<15} {:<20}".format("ID", "Last Seen Camera", "Last Seen Time"))
    print("-" * 50)

    for i, row in enumerate(tracked_data[:10]):  # Slice to get the first 10 rows
        person_id, last_seen_cam, last_seen_time = row
        # Convert the timestamp to a readable time string
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_seen_time))
        print("{:<10} {:<15} {}".format(person_id, last_seen_cam, time_str))
        
    print("\n=================================")



if __name__ == "__main__":
    video_sources = ["feed3.mp4", "feed4.mp4"]
    manager = Manager()
    shared_selected_ids = manager.list()
    processes = []
    for i, src in enumerate(video_sources):
        p = Process(target=process_video, args=(src, i, shared_selected_ids))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    cv2.destroyAllWindows()

    display_tracking_data()

