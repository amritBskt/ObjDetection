# Multi-Camera Person Tracking System

## Overview
This project implements a multi-camera person tracking system using YOLOv9 for object detection, DeepSORT for tracking, and an OSNet-based ReID model for person re-identification. The system also incorporates an SQLite database for logging and tracking person identities across multiple video feeds.

## Features
- **Multi-Camera Support**: Track individuals across multiple video sources.
- **Person Detection**: Uses YOLOv9 for accurate person detection.
- **Person Tracking**: Implements DeepSORT for robust object tracking.
- **Re-Identification (ReID)**: Utilizes OSNet for feature extraction and cosine similarity for identity matching.
- **Manual Selection**: Allows users to click on detected persons for tracking adjustments.
- **SQLite Database Integration**: Stores and updates person tracking data.
- **Logging**: Debugging and event logging using Python's `logging` module.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install ultralytics opencv-python numpy torch torchvision scipy deep-sort-realtime
```

Additionally, install `torchreid-1.4.0` from the following repository:

```bash
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -r requirements.txt
python setup.py install
```

### Database Initialization
Before running the tracker, ensure the SQLite database is initialized:

```python
python -c 'import sqlite3; conn = sqlite3.connect("reid_db.sqlite3"); cursor = conn.cursor(); cursor.execute("CREATE TABLE IF NOT EXISTS persons (id INTEGER PRIMARY KEY, feature BLOB, last_seen_camera INTEGER, last_seen_time REAL)"); conn.commit(); conn.close()'
```

## Usage
Run the script with multiple video sources:

```bash
python main.py
```

## Configuration
- **YOLO Model**: The script uses `yolov9s.pt` but can be changed to other models.
- **DeepSORT Parameters**: `max_age=30, nn_budget=100` (Adjust for tracking stability)
- **ReID Model**: `osnet_x1_ain_0.pth` (Ensure the model file is in the working directory)
- **Tracking Threshold**: Set in `find_matching_person(new_feature, cam_id, threshold=0.7)`

## How It Works
1. **Load Models**: YOLOv9, DeepSORT, and OSNet ReID are initialized.
2. **Process Video Streams**: Each camera feed is handled in a separate process.
3. **Detect & Track**: YOLO detects persons, DeepSORT assigns IDs, and OSNet extracts deep features.
4. **Person Matching**: Compares features using cosine similarity and updates the database.
5. **User Interaction**: Click on detected persons to mark/unmark them.
6. **Database Logging**: Tracks last seen camera and timestamp.
7. **Tracking Data Display**: Prints tracked person details upon completion.

## Example Output
```plaintext
[INFO] Processing video: video1.mp4
[INFO] Processing video: video2.mp4
[INFO] ReID Model Loaded Successfully!
[INFO] Selected Person Data: (ID, Feature Vector, Camera, Timestamp)
===== Final Tracked Persons =====
ID        Last Seen Camera  Last Seen Time
------------------------------------------
1         0                2025-02-18 15:30:45
2         1                2025-02-18 15:31:12
==========================================
```

## Troubleshooting
- **Issue: CUDA not available**
  - Ensure PyTorch is installed with GPU support.
- **Issue: Video not opening**
  - Check if the file path is correct.
- **Issue: Low tracking accuracy**
  - Tune `max_age`, `nn_budget`, or `threshold` parameters.

## Future Improvements
- Add GUI for easier manual tracking adjustments.
- Implement a more advanced database for scalability.
- Optimize multi-camera synchronization.

## Sample Input Video
![Demo](inputSample.gif)

## Sample Output Video
![Demo](outputSample.gif)


