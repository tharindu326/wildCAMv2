#!/usr/bin/env python3
"""
Camera Manager for ArduCam Quad Camera Kit on Raspberry Pi 5.

The ArduCam Quad Camera module presents all 4 cameras as a SINGLE libcamera
device via the CamArray HAT. In the default 4-in-1 composition mode, the
output is one combined frame with all 4 camera views in a 2x2 grid:

    ┌──────┬──────┐
    │ Cam0 │ Cam1 │
    ├──────┼──────┤
    │ Cam2 │ Cam3 │
    └──────┴──────┘

This module:
  1. Opens a single Picamera2 instance (device 0 = the combined quad output)
  2. Captures the combined 4-in-1 frame
  3. Splits it into 4 quadrants
  4. Resizes each quadrant to the target per-camera resolution
  5. Serves each quadrant as a separate "camera" to the rest of the pipeline
"""

import cv2
import threading
import time
import logging
import numpy as np
from config import cfg

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Quadrant layout for the 2x2 grid (cam_id -> grid position)
# The CamArray HAT stitches cameras in this order:
#   [0] [1]
#   [2] [3]
QUADRANT_MAP = {
    0: (0, 0),  # top-left
    1: (0, 1),  # top-right
    2: (1, 0),  # bottom-left
    3: (1, 1),  # bottom-right
}


class CameraManager:
    """
    Manager for ArduCam Quad Camera Kit (4-in-1 composition mode).

    Captures from a single Picamera2 device, splits the combined frame
    into 4 camera quadrants, and serves them independently.
    """

    def __init__(self, num_cameras: int = None):
        """
        Initialize camera manager.

        Args:
            num_cameras: Number of cameras in the quad kit (1-4, default from config).
        """
        self.num_cameras = num_cameras or cfg.camera.num_cameras
        self.picam2 = None
        self.camera_ids = []
        self.running = False

        # Latest frames per camera (quadrant)
        self._frames = {}
        self._frame_ids = {}
        self._timestamps = {}
        self._lock = threading.Lock()
        self._thread = None

        # Per-camera target resolution (what the pipeline receives)
        self.target_width = cfg.camera.frame_width    # 640
        self.target_height = cfg.camera.frame_height  # 480

        # Combined (4-in-1) frame resolution sent to Picamera2
        # For the ArduCam 64MP quad kit, available sensor modes include:
        #   1280x720 @120fps  → 640x360 per cam
        #   1920x1080 @60fps  → 960x540 per cam
        #   2312x1736 @30fps  → 1156x868 per cam
        # Default: 1920x1080 gives 960x540 per quadrant (good quality + 60fps headroom)
        self.combined_width = getattr(cfg.camera, 'combined_width', 1920)
        self.combined_height = getattr(cfg.camera, 'combined_height', 1080)

    def initialize_cameras(self, camera_ids: list = None):
        """
        Initialize the ArduCam Quad Camera.

        Args:
            camera_ids: List of logical camera IDs to use (subset of [0,1,2,3]).
                        Each maps to a quadrant in the 2x2 grid.
                        Default: [0, 1, 2, 3] for all 4 cameras.
        """
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "picamera2 is not installed. Install with:\n"
                "  sudo apt install -y python3-picamera2"
            )

        if camera_ids is None:
            camera_ids = list(range(self.num_cameras))

        # Validate IDs against the 2x2 quadrant layout
        for cid in camera_ids:
            if cid not in QUADRANT_MAP:
                raise ValueError(
                    f"Camera ID {cid} invalid. ArduCam Quad Kit supports IDs 0-3 "
                    f"(2x2 grid quadrants)."
                )

        self.camera_ids = camera_ids

        # Open the single combined device (always device 0)
        try:
            self.picam2 = Picamera2(camera_num=0)
        except Exception as e:
            raise RuntimeError(f"Failed to open ArduCam Quad Camera (device 0): {e}")

        # Configure for BGR output at the combined resolution
        config = self.picam2.create_preview_configuration(
            main={
                "size": (self.combined_width, self.combined_height),
                "format": "BGR888"
            },
            controls={
                "FrameRate": cfg.camera.fps
            }
        )
        self.picam2.configure(config)

        # Initialize per-camera frame storage
        for cam_id in self.camera_ids:
            self._frames[cam_id] = None
            self._frame_ids[cam_id] = 0
            self._timestamps[cam_id] = 0

        logger.info(
            f"ArduCam Quad Camera initialized: "
            f"combined={self.combined_width}x{self.combined_height}, "
            f"per-camera target={self.target_width}x{self.target_height}, "
            f"cameras={self.camera_ids}"
        )

    def start_all(self):
        """Start the camera and begin capturing/splitting frames."""
        self.picam2.start()

        # Warmup: let auto-exposure and auto-white-balance settle
        warmup = getattr(cfg.camera, 'warmup_seconds', 1.0)
        logger.info(f"Camera warming up for {warmup}s (AE/AWB settling)...")
        time.sleep(warmup)

        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Quad camera capture started")

    def _capture_loop(self):
        """Continuously capture the combined frame, split into quadrants."""
        while self.running:
            try:
                # Capture the full 4-in-1 combined frame
                combined = self.picam2.capture_array("main")
                h, w = combined.shape[:2]
                mid_x = w // 2
                mid_y = h // 2

                # Split into quadrants
                quadrants = {
                    0: combined[0:mid_y, 0:mid_x],       # top-left
                    1: combined[0:mid_y, mid_x:w],        # top-right
                    2: combined[mid_y:h, 0:mid_x],        # bottom-left
                    3: combined[mid_y:h, mid_x:w],         # bottom-right
                }

                now = time.time()

                with self._lock:
                    for cam_id in self.camera_ids:
                        quad = quadrants[cam_id]

                        # Resize to target per-camera resolution if needed
                        qh, qw = quad.shape[:2]
                        if qw != self.target_width or qh != self.target_height:
                            quad = cv2.resize(
                                quad,
                                (self.target_width, self.target_height),
                                interpolation=cv2.INTER_LINEAR
                            )

                        self._frames[cam_id] = quad
                        self._frame_ids[cam_id] += 1
                        self._timestamps[cam_id] = now

            except Exception as e:
                logger.warning(f"Capture error: {e}")
                time.sleep(0.01)

    def stop_all(self):
        """Stop the camera capture and release resources."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        logger.info("All cameras stopped")

    def read_frame(self, camera_id: int):
        """
        Read the latest frame from a specific camera quadrant.

        Args:
            camera_id: Camera ID (0-3).

        Returns:
            tuple: (success, frame, frame_id, timestamp)
        """
        with self._lock:
            frame = self._frames.get(camera_id)
            if frame is None:
                return False, None, 0, 0
            return (
                True,
                frame.copy(),
                self._frame_ids[camera_id],
                self._timestamps[camera_id]
            )

    def read_all_frames(self):
        """
        Read latest frames from all active camera quadrants.

        Returns:
            dict: {camera_id: (success, frame, frame_id, timestamp)}
        """
        frames = {}
        with self._lock:
            for cam_id in self.camera_ids:
                frame = self._frames.get(cam_id)
                if frame is None:
                    frames[cam_id] = (False, None, 0, 0)
                else:
                    frames[cam_id] = (
                        True,
                        frame.copy(),
                        self._frame_ids[cam_id],
                        self._timestamps[cam_id]
                    )
        return frames

    def get_camera_info(self, camera_id: int):
        """
        Get camera information.

        Args:
            camera_id: Camera ID (0-3).

        Returns:
            dict: Camera properties, or None if camera_id is invalid.
        """
        if camera_id not in self.camera_ids:
            return None

        return {
            'camera_id': camera_id,
            'width': self.target_width,
            'height': self.target_height,
            'fps': cfg.camera.fps,
            'running': self.running
        }

    def get_active_camera_ids(self):
        """Get list of active camera IDs."""
        if self.running:
            return list(self.camera_ids)
        return []

    def __enter__(self):
        """Context manager entry."""
        self.initialize_cameras()
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_all()
        return False


def test_cameras():
    """Test: capture from the quad camera and display all 4 quadrants."""
    logger.info("Testing ArduCam Quad Camera Manager...")

    try:
        manager = CameraManager()
        manager.initialize_cameras([0, 1, 2, 3])
        manager.start_all()

        logger.info("Capturing test frames for 5 seconds...")
        start = time.time()
        frame_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        while time.time() - start < 5.0:
            frames = manager.read_all_frames()
            for cam_id, (success, frame, fid, ts) in frames.items():
                if success:
                    frame_counts[cam_id] = fid
            time.sleep(1 / 30)

        elapsed = time.time() - start
        for cam_id, count in frame_counts.items():
            fps = count / elapsed if elapsed > 0 else 0
            logger.info(f"Camera {cam_id}: {count} frames in {elapsed:.1f}s ({fps:.1f} fps)")

        manager.stop_all()
        logger.info("Camera test complete")

    except Exception as e:
        logger.error(f"Camera test failed: {e}")


if __name__ == '__main__':
    test_cameras()
