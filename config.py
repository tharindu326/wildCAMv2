#!/usr/bin/env python
import os
from easydict import EasyDict as edict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

__C = edict()
cfg = __C

# Detector configuration
__C.detector = edict()
__C.detector.weight_file = "model_data/yolov8n_192_full_integer_quant_edgetpu.tflite"
__C.detector.classes = [0]  # filter by class: --class 0, or --class 0 2 3
__C.detector.OBJECTNESS_CONFIDANCE = 0.3  # Minimum confidence for detection
__C.detector.NMS_THRESHOLD = 0.45  # Non-maximum suppression threshold
__C.detector.device = 'cpu'  # 'cpu' or GPU device ID
__C.detector.verbose = False

# Detection filtering
__C.filter = edict()
__C.filter.NMS_THRESHOLD = 0.7
__C.filter.image_size_factor = 2000  # Min box area = image_area / this factor
__C.filter.min_box_area_adjust = 0  # Additional adjustment for min box area

# Display flags
__C.flags = edict()
__C.flags.image_show = False  # Show preview window (disabled for headless RPi)
__C.flags.render_detections = True  # Draw detection overlays
__C.flags.render_labels = True  # Show class labels
__C.flags.render_fps = True  # Show FPS counter

# Video recording settings
__C.video = edict()
__C.video.output_path = 'output/'  # Directory for recordings
__C.video.video_writer_fps = 30  # Output video FPS
__C.video.FOURCC = 'mp4v'  # Video codec
__C.video.save = True  # Enable video recording
__C.video.recording_duration = 60  # Max recording duration in seconds

# Camera configuration
__C.camera = edict()
__C.camera.num_cameras = 4  # Number of ArduCams connected
__C.camera.frame_width = 640  # Camera frame width
__C.camera.frame_height = 480  # Camera frame height
__C.camera.fps = 30  # Camera capture FPS

# Recording candidate settings
__C.recording = edict()
__C.recording.confirmation_frames = 5  # Frames a track must appear to confirm (FP filter)
__C.recording.disappear_frames = 10  # Frames a track must be missing before removal
__C.recording.min_bbox_ratio = 0.02  # Min bbox area ratio to frame for video recording (2%)
__C.recording.photo_interval = 1.0  # Seconds between photo captures when bbox too small
__C.recording.max_video_duration = 300  # Max video duration in seconds (5 min)
__C.recording.frame_buffer_size = 90  # Frames to buffer (3 seconds at 30fps)

# General settings
__C.general = edict()
__C.general.frame_rotate = False  # Rotate frame 90 degrees
__C.general.frame_resize = None  # Resize frame to (width, height) or None
__C.general.GPSlocation = '60.4575N-24.9588E'  # GPS location for metadata
__C.general.COLORS = {
                          'green': [64, 255, 64],
                          'blue': [255, 128, 0],
                          'coral': [0, 128, 255],
                          'yellow': [0, 255, 255],
                          'gray': [169, 169, 169],
                          'cyan': [255, 255, 0],
                          'magenta': [255, 0, 255],
                          'white': [255, 255, 255],
                          'red': [64, 0, 255],
                          'candidate_red': [0, 0, 255]  # BGR for candidate bbox
                      }

# Trackers
__C.tracker = edict()
__C.tracker.type = 'bytetrack'  # Only using bytetrack
__C.tracker.classes = []  # classes id to track (empty = all classes)
__C.tracker.reid_weights = 'model_data/osnet_x0_25_msmt17.pt'
__C.tracker.time_since_update_threshold = 6
__C.tracker.trail_length = 60
__C.tracker.enable = True

# ByteTracker: In use
__C.bytetrack = edict()
__C.bytetrack.track_thresh = 0.2  # if the confidence_score> track_thresh + det_tresh_gap then initialize a new track otherwise it will only match tracklets where confidance_score> track_thresh
__C.bytetrack.track_buffer = 30  # length of maximum frames where can a lost tracklet be. if the tracklet not appear within 30 frames track will be deleted., else track will rebirth
__C.bytetrack.match_thresh = 0.95  # linear assignment threshold where it uses Jonker-Volgenant algorithm. when this is lower no tracklets age. this can be also defined as cost of assignment of the Jonker-Volgenant algorithm. maximum error that allow for the linear assignment.
__C.bytetrack.frame_rate = 30  # frame rate of the video; used to define the buffer size; buffer_size = int(frame_rate / 30.0 * self.track_buffer)


# OCSORT and Strong sort are not in use. They are just optional trackers, we are only using ByteTrack
# StrongSort
__C.strongsort = edict()
__C.strongsort.ecc = True
__C.strongsort.ema_alpha = 0.8962157769329083
__C.strongsort.max_age = 40
__C.strongsort.max_dist = 0.1594374041012136
__C.strongsort.max_iou_dist = 0.5431835667667874
__C.strongsort.max_unmatched_preds = 0
__C.strongsort.mc_lambda = 0.995
__C.strongsort.n_init = 3
__C.strongsort.nn_budget = 100
__C.strongsort.conf_thres = 0.5122620708221085

# OCSort
__C.ocsort = edict()
__C.ocsort.asso_func = 'giou'
__C.ocsort.conf_thres = 0.5122620708221085
__C.ocsort.delta_t = 1
__C.ocsort.det_thresh = 0
__C.ocsort.inertia = 0.3941737016672115
__C.ocsort.iou_thresh = 0.22136877277096445
__C.ocsort.max_age = 50
__C.ocsort.min_hits = 1
__C.ocsort.use_byte = False

# BoostTrack

__C.boosttrack = edict()
__C.boosttrack.det_thresh = 0.2
__C.boosttrack.lambda_iou = 0.2
__C.boosttrack.lambda_mhd = 0.25
__C.boosttrack.lambda_shape = 0.25
__C.boosttrack.dlo_boost_coef = 0.65
__C.boosttrack.use_dlo_boost = True
__C.boosttrack.use_duo_boost = True
__C.boosttrack.max_age = 30


# S3 Upload Configuration
__C.s3 = edict()
__C.s3.enable = True  # Enable S3 upload
# Load from environment variables (from .env file or system)
__C.s3.region = os.getenv('AWS_REGION', 'eu-central-1')
__C.s3.access_key_id = os.getenv('AWS_ACCESS_KEY_ID', '')
__C.s3.secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', '')
__C.s3.bucket_name = os.getenv('AWS_BUCKET_NAME', 'bigfoot2025')
__C.s3.delete_local_after_upload = True  # Delete local files after successful upload
__C.s3.ExpiresIn = 604800  # URL expiration time in seconds (7 days)

# Verify credentials are loaded
if not __C.s3.access_key_id or not __C.s3.secret_access_key:
    print("⚠️ WARNING: AWS credentials not found in environment variables!")
    print("Make sure .env file exists or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")