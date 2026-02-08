# Wildlife Camera Detection & Recording System

A multi-camera wildlife monitoring system for Raspberry Pi 5 with ArduCam support. The system continuously captures frames from multiple cameras, detects animals using YOLO, tracks them across frames with ByteTrack, and intelligently records videos/photos which are uploaded to AWS S3.

## Table of Contents

- [System Overview](#system-overview)
- [Installation](#installation)
- [Usage](#usage)
- [System Behavior](#system-behavior)
  - [1. Frame Capture](#1-frame-capture)
  - [2. Detection Pipeline](#2-detection-pipeline)
  - [3. Tracking Pipeline](#3-tracking-pipeline)
  - [4. Recording Logic](#4-recording-logic)
  - [5. Visual Output](#5-visual-output)
  - [6. S3 Upload Pipeline](#6-s3-upload-pipeline)
  - [7. Complete Data Flow](#7-complete-data-flow)
  - [8. Example Scenario Timeline](#8-example-scenario-timeline)
- [Configuration](#configuration)
- [Output Files Structure](#output-files-structure)
- [Project Structure](#project-structure)

---

## System Overview

The system runs on a Raspberry Pi 5 with 4 ArduCams connected. It continuously captures frames from all cameras, detects animals, tracks them across frames, and intelligently records videos/photos which are uploaded to S3.

**Key Features:**
- Multi-camera support (4 ArduCams)
- Real-time animal detection using YOLO
- Object tracking with ByteTrack (persistent IDs across frames)
- False positive filtering (confirmation threshold)
- Intelligent video vs photo decision based on bbox size
- Automatic video spawning for individual tracks
- Background S3 uploads (non-blocking)

---

## Installation

### Requirements

- Raspberry Pi 5
- 4x ArduCam cameras
- Python 3.8+

### Setup

```bash
# Clone or copy the project
cd wildCAMv2

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials (create .env file)
cat > .env << EOF
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=eu-central-1
AWS_BUCKET_NAME=your-bucket-name
EOF
```

---

## Usage

```bash
# Run with all 4 cameras (default)
python main.py

# Run with specific cameras (by index)
python main.py --cameras 0 1 2 3

# Raspberry Pi with 4 CSI / Pi cameras: use even indices (0 2 4 6) or device paths.
# Odd /dev/video nodes (1, 3, 5...) are often metadata, not capture devices.
python main.py --cameras 0 2 4 6
# Or explicitly:
python main.py --cameras /dev/video0 /dev/video2 /dev/video4 /dev/video6

# Single camera test mode with preview
python main.py --single-camera 0 --show

# Test on video file
python main.py --video path/to/video.mp4 --show

# Disable S3 uploads
python main.py --no-s3

# Enable debug logging
python main.py --debug
```

---

## System Behavior

### 1. Frame Capture

```
┌─────────────────────────────────────────────────────────┐
│                    CAMERA MANAGER                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Camera 0 │ │ Camera 1 │ │ Camera 2 │ │ Camera 3 │   │
│  │ Thread   │ │ Thread   │ │ Thread   │ │ Thread   │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │
│       │            │            │            │          │
│       ▼            ▼            ▼            ▼          │
│   [Frame Buffer] [Frame Buffer] [Frame Buffer] [Frame Buffer]
└─────────────────────────────────────────────────────────┘
```

- Each camera runs in its own **background thread**
- Continuously captures frames at 30 FPS
- Frames are stored in a thread-safe buffer
- Main loop reads the latest frame from each camera without blocking

---

### 2. Detection Pipeline

For each frame from each camera:

```
Frame → YOLO Detection → Filter Overlapping Boxes → Filter Small Boxes → Detections
```

**Output format:** `[x, y, width, height]` for each detected animal

---

### 3. Tracking Pipeline

```
Detections → ByteTrack → Tracked Objects with IDs
```

**Output format:** `[x1, y1, x2, y2, track_id, class_id, confidence]`

- Each animal gets a **unique track_id** that persists across frames
- ByteTrack uses Kalman filter for motion prediction
- Can re-identify animals after brief occlusions (up to 30 frames)

---

### 4. Recording Logic

This is the core logic. Here's the complete flow:

#### Step 1: Track Management

```
For each tracked object in frame:
    │
    ├─► New track_id?
    │       └─► Create TrackCandidate (frames_seen=1, is_confirmed=False)
    │
    └─► Existing track_id?
            └─► Update: frames_seen++, frames_missing=0, update bbox
```

#### Step 2: False Positive (FP) Filtering

```
if track.frames_seen >= 5 (confirmation_frames):
    track.is_confirmed = True
    └─► "Track 42 confirmed after 5 frames"
```

**Why?** Random false detections won't persist for 5+ consecutive frames. Real animals will.

#### Step 3: Bbox Size Check (Video vs Photo Decision)

```
bbox_ratio = (bbox_width × bbox_height) / (frame_width × frame_height)

if bbox_ratio >= 0.02 (2% of frame):
    └─► RECORD VIDEO (animal is large enough, good visibility)
else:
    └─► CAPTURE PHOTOS every 1 second (animal too small/far)
```

**Why?** Small distant animals don't make good video. Photos are sufficient and save storage.

#### Step 4: Recording Session Management

```
┌────────────────────────────────────────────────────────────────┐
│                     RECORDING STATE MACHINE                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NO CANDIDATES ──────► CANDIDATES APPEAR ──────► START RECORDING
│       ▲                                               │
│       │                                               ▼
│       │                                    WRITE FRAMES WITH
│       │                                    RED BBOXES DRAWN
│       │                                               │
│  STOP RECORDING ◄──── ALL CANDIDATES GONE ◄──────────┘
│       │
│       ▼
│  FINALIZE & UPLOAD
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

#### Step 5: Multiple Animals Handling

```
Recording Session Active with Track 42
    │
    ├─► New Track 57 appears (large bbox, confirmed)
    │       └─► ADD to existing session (don't start new recording)
    │       └─► Both Track 42 and 57 get red bboxes in video
    │
    └─► Track 99 appears (small bbox)
            └─► Only capture photos for Track 99
            └─► Video continues with 42 and 57
```

#### Step 6: Track Disappearance Handling

```
Track 42 not seen in current frame:
    │
    └─► track.frames_missing++
        │
        └─► if frames_missing >= 10 (disappear_frames):
                │
                ├─► SPAWN separate video for Track 42
                │   (from when it first appeared to now)
                │
                ├─► REMOVE Track 42 from recording session
                │
                └─► Continue recording with remaining candidates (Track 57)
```

#### Step 7: Video Spawning (The Tricky Part)

**Problem:** How to create a video for Track 42 without stopping the main recording?

**Solution:** Frame buffering per track

```
┌─────────────────────────────────────────────────────────────┐
│                    FRAME STORAGE                             │
│                                                              │
│  Main Video Writer ──────► wildlife_cam0_20240125_143022.mp4│
│       │                                                      │
│       │ (writes every frame with all candidate bboxes)       │
│       │                                                      │
│  Track 42 Buffer ────────► Stores copies of frames          │
│  Track 57 Buffer ────────► Stores copies of frames          │
│                                                              │
│  When Track 42 ends:                                         │
│       └─► Write Track 42 Buffer → track_42_cam0_....mp4     │
│       └─► Upload to S3 in background thread                  │
│       └─► Clear Track 42 Buffer                              │
│       └─► Main recording continues uninterrupted             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 5. Visual Output

Each frame in the video shows:

```
┌────────────────────────────────────────────────┐
│ Cam0 | FPS: 28.5 | Frame: 1542                 │
│ REC | Tracks: 2                                │
│                                                │
│         ┌─────────────┐                        │
│         │ ID:42       │ ◄── RED bbox (thick)   │
│         │             │     for confirmed      │
│         │   [deer]    │     recording candidate│
│         │             │                        │
│         └─────────────┘                        │
│                                                │
│              ┌───────┐                         │
│              │ID:57  │ ◄── RED bbox            │
│              │ [fox] │                         │
│              └───────┘                         │
│                                                │
└────────────────────────────────────────────────┘
```

---

### 6. S3 Upload Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    UPLOAD PIPELINE                           │
│                                                              │
│  Video Completed ──► Upload Queue ──► Background Thread     │
│  Photo Captured  ──►      │                                 │
│                          │                                  │
│                          ▼                                  │
│                   ┌─────────────┐                           │
│                   │ S3 Manager  │                           │
│                   │   Thread    │                           │
│                   └──────┬──────┘                           │
│                          │                                  │
│                          ▼                                  │
│         s3://bucket/recordings/cam0/video_xxx.mp4           │
│         s3://bucket/recordings/cam0/photos/photo_xxx.jpg    │
│                          │                                  │
│                          ▼                                  │
│              Delete local file after upload                 │
│              (if cfg.s3.delete_local_after_upload = True)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key:** Uploads happen in a separate thread, so the main detection loop never blocks.

---

### 7. Complete Data Flow

Per camera data flow:

```
                          CAMERA 0
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  CAPTURE        │  DETECT         │  TRACK        │ RECORD  │
│  ───────        │  ──────         │  ─────        │ ──────  │
│                 │                 │               │         │
│  ArduCam ──►    │  YOLO ──►       │  ByteTrack──► │ Manager │
│  Thread         │  Inference      │  Tracker      │         │
│       │         │       │         │       │       │    │    │
│       ▼         │       ▼         │       ▼       │    ▼    │
│    Frame        │   Boxes +       │  Tracked      │ Video/  │
│    Buffer       │   Confidence    │  Objects      │ Photos  │
│                 │                 │  with IDs     │    │    │
└─────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
                                                   S3 Upload
                                                    Thread
```

---

### 8. Example Scenario Timeline

```
Frame 0-4:    Track 42 detected (deer) - NOT YET CONFIRMED
              → No recording, waiting for confirmation

Frame 5:      Track 42 confirmed! (seen 5 consecutive frames)
              → Bbox is 5% of frame (> 2% threshold)
              → START RECORDING SESSION
              → Draw red bbox on Track 42

Frame 6-50:   Track 42 continues
              → Recording continues
              → Each frame written to video

Frame 51:     Track 57 appears (fox), small bbox (1% of frame)
              → Capture photo (bbox too small for video)
              → Track 57 NOT added to video session

Frame 56:     Track 57 confirmed + bbox grows to 3%
              → ADD Track 57 to recording session
              → Now both 42 and 57 have red bboxes in video

Frame 100:    Track 42 not detected (deer left frame)

Frame 101-109: Track 42 still missing
               → frames_missing counting up

Frame 110:    Track 42 missing for 10 frames
              → SPAWN video for Track 42 (frames 5-100)
              → REMOVE Track 42 from session
              → UPLOAD track_42 video to S3 (background)
              → Continue recording with Track 57 only

Frame 200:    Track 57 disappears for 10 frames
              → SPAWN video for Track 57
              → No more candidates
              → STOP RECORDING SESSION
              → UPLOAD main video to S3
              → UPLOAD track_57 video to S3
```

---

## Configuration

All configuration is in `config.py`. Key parameters:

### Recording Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recording.confirmation_frames` | 5 | Frames before track is confirmed (FP filter) |
| `recording.disappear_frames` | 10 | Frames missing before track removal |
| `recording.min_bbox_ratio` | 0.02 | Min bbox ratio for video (below = photos) |
| `recording.photo_interval` | 1.0 | Seconds between photo captures |
| `recording.max_video_duration` | 300 | Max video length in seconds (5 min) |
| `recording.frame_buffer_size` | 90 | Frames to buffer for track video spawning |

### Camera Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera.num_cameras` | 4 | Number of ArduCams connected |
| `camera.frame_width` | 640 | Camera frame width |
| `camera.frame_height` | 480 | Camera frame height |
| `camera.fps` | 30 | Camera capture FPS |
| `camera.warmup_frames` | 5 | Frames read after open to stabilize driver (avoids Pi segfaults) |

**Raspberry Pi + CSI cameras:** Use `--cameras 0 2 4 6` or `--cameras /dev/video0 /dev/video2 /dev/video4 /dev/video6`. Check capture devices with `v4l2-ctl --list-devices` or `libcamera-hello --list-cameras`.

### Detection Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detector.weight_file` | yolov8n_192...tflite | Model file path |
| `detector.OBJECTNESS_CONFIDANCE` | 0.3 | Min detection confidence |
| `detector.NMS_THRESHOLD` | 0.45 | Non-max suppression threshold |

### Tracker Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tracker.type` | bytetrack | Tracker algorithm |
| `bytetrack.track_buffer` | 30 | Frames to keep lost tracks |
| `bytetrack.track_thresh` | 0.2 | Detection confidence threshold |

### S3 Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `s3.enable` | True | Enable S3 uploads |
| `s3.delete_local_after_upload` | True | Delete local files after upload |
| `s3.bucket_name` | (from .env) | S3 bucket name |

---

## Output Files Structure

```
output/
├── wildlife_cam0_20240125_143022.mp4    # Main recording (all candidates)
├── wildlife_cam0_20240125_143523.mp4    # Another recording session
├── track_42_cam0_20240125_143022.mp4    # Spawned video for track 42
├── track_57_cam0_20240125_143156.mp4    # Spawned video for track 57
└── photos/
    ├── photo_track99_cam0_20240125_143044.jpg
    ├── photo_track99_cam0_20240125_143045.jpg
    └── photo_track99_cam0_20240125_143046.jpg
```

After S3 upload (with `delete_local_after_upload=True`), local files are removed.

S3 structure:
```
s3://bucket-name/
└── recordings/
    ├── cam0/
    │   ├── wildlife_cam0_20240125_143022.mp4
    │   ├── track_42_cam0_20240125_143022.mp4
    │   └── photos/
    │       └── photo_track99_cam0_20240125_143044.jpg
    ├── cam1/
    ├── cam2/
    └── cam3/
```

---

## Project Structure

```
wildCAMv2/
├── main.py                 # Main entry point
├── config.py               # Configuration settings
├── inference.py            # YOLO detection
├── tracker.py              # ByteTrack wrapper
├── camera_manager.py       # Multi-camera handling
├── recording_manager.py    # Recording logic
├── s3_manager.py           # AWS S3 uploads
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env                    # AWS credentials (create this)
├── model_data/             # Detection models
│   ├── yolov8n_192_full_integer_quant_edgetpu.tflite
│   └── ...
├── trackers/               # Tracking algorithms
│   ├── bytetrack/
│   ├── strongsort/
│   ├── ocsort/
│   └── boosttrack/
└── output/                 # Generated videos/photos
    └── photos/
```

---

## License

[Add your license here]

---

## Acknowledgments

- YOLO/Ultralytics for object detection
- ByteTrack for multi-object tracking
- ArduCam for Raspberry Pi camera modules
