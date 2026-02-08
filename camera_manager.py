#!/usr/bin/env python3
"""
Camera Manager for ArduCam Quad Camera Kit on Raspberry Pi 5
Uses Picamera2 (libcamera) for CSI camera capture with threading support.

The ArduCam Quad Camera module connects 4 cameras via a single CSI ribbon cable.
After ArduCam driver installation, all 4 cameras appear as libcamera devices 0-3.
"""

import cv2
import threading
import time
import logging
from config import cfg

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraStream:
    """Thread-safe camera stream handler for a single Pi camera via Picamera2"""

    def __init__(self, camera_num: int, width: int = None, height: int = None, fps: int = None):
        """
        Initialize camera stream.

        Args:
            camera_num: Libcamera device index (0-3 for ArduCam Quad Kit).
            width: Frame width (default from config).
            height: Frame height (default from config).
            fps: Target FPS (default from config).
        """
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "picamera2 is not installed. Install it with: sudo apt install -y python3-picamera2"
            )

        self.camera_num = camera_num
        self.width = width or cfg.camera.frame_width
        self.height = height or cfg.camera.frame_height
        self.fps = fps or cfg.camera.fps

        self.picam2 = None
        self.frame = None
        self.frame_id = 0
        self.timestamp = 0
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

        self.actual_width = self.width
        self.actual_height = self.height
        self.actual_fps = self.fps

        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize the Picamera2 instance and configure it."""
        try:
            self.picam2 = Picamera2(camera_num=self.camera_num)
        except Exception as e:
            raise RuntimeError(f"Failed to open camera {self.camera_num}: {e}")

        # Configure for BGR output so frames are directly compatible with OpenCV
        config = self.picam2.create_preview_configuration(
            main={
                "size": (self.width, self.height),
                "format": "BGR888"
            },
            controls={
                "FrameRate": self.fps
            }
        )
        self.picam2.configure(config)

        logger.info(
            f"Camera {self.camera_num} initialized: {self.width}x{self.height} @ {self.fps}fps (Picamera2)"
        )

    def start(self):
        """Start the camera capture thread."""
        if self.running:
            return self

        # Start the camera hardware
        self.picam2.start()

        # Warmup: let the auto-exposure and auto-white-balance settle
        warmup_time = getattr(cfg.camera, "warmup_seconds", 0.5)
        logger.info(f"Camera {self.camera_num}: warming up for {warmup_time}s...")
        time.sleep(warmup_time)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        """Continuous frame capture loop."""
        while self.running:
            try:
                # capture_array returns a numpy array in BGR888 format (OpenCV-compatible)
                frame = self.picam2.capture_array("main")
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                    self.timestamp = time.time()
            except Exception as e:
                logger.warning(f"Camera {self.camera_num}: capture error: {e}")
                time.sleep(0.01)

    def read(self):
        """
        Read the latest frame.

        Returns:
            tuple: (success, frame, frame_id, timestamp)
        """
        with self.lock:
            if self.frame is None:
                return False, None, 0, 0
            return True, self.frame.copy(), self.frame_id, self.timestamp

    def stop(self):
        """Stop the camera capture and release resources."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        logger.info(f"Camera {self.camera_num} stopped")

    def is_running(self):
        """Check if camera is running."""
        return self.running and self.picam2 is not None


class CameraManager:
    """Manager for multiple Picamera2 camera streams (ArduCam Quad Kit)"""

    def __init__(self, num_cameras: int = None):
        """
        Initialize camera manager.

        Args:
            num_cameras: Number of cameras to manage (default from config).
        """
        self.num_cameras = num_cameras or cfg.camera.num_cameras
        self.cameras = {}
        self.running = False

    def initialize_cameras(self, camera_ids: list = None):
        """
        Initialize all cameras.

        Args:
            camera_ids: List of libcamera device indices (e.g. [0, 1, 2, 3]).
                        Defaults to [0, 1, ..., num_cameras-1].
        """
        if camera_ids is None:
            camera_ids = list(range(self.num_cameras))

        for cam_id in camera_ids:
            # Ensure int index for Picamera2
            cam_num = int(cam_id) if not isinstance(cam_id, int) else cam_id
            try:
                camera = CameraStream(cam_num)
                self.cameras[cam_num] = camera
                logger.info(f"Initialized camera {cam_num}")
            except RuntimeError as e:
                logger.error(f"Failed to initialize camera {cam_num}: {e}")

        if not self.cameras:
            raise RuntimeError("No cameras could be initialized")

        logger.info(f"Initialized {len(self.cameras)} cameras")

    def start_all(self):
        """Start all camera streams."""
        self.running = True
        for cam_id, camera in self.cameras.items():
            camera.start()
            logger.info(f"Started camera {cam_id}")

    def stop_all(self):
        """Stop all camera streams."""
        self.running = False
        for cam_id, camera in self.cameras.items():
            camera.stop()
        logger.info("All cameras stopped")

    def read_frame(self, camera_id: int):
        """
        Read frame from a specific camera.

        Args:
            camera_id: Camera ID to read from.

        Returns:
            tuple: (success, frame, frame_id, timestamp)
        """
        if camera_id not in self.cameras:
            return False, None, 0, 0
        return self.cameras[camera_id].read()

    def read_all_frames(self):
        """
        Read frames from all cameras.

        Returns:
            dict: {camera_id: (success, frame, frame_id, timestamp)}
        """
        frames = {}
        for cam_id, camera in self.cameras.items():
            frames[cam_id] = camera.read()
        return frames

    def get_camera_info(self, camera_id: int):
        """
        Get camera information.

        Args:
            camera_id: Camera ID.

        Returns:
            dict: Camera properties.
        """
        if camera_id not in self.cameras:
            return None

        camera = self.cameras[camera_id]
        return {
            'camera_id': camera_id,
            'width': camera.actual_width,
            'height': camera.actual_height,
            'fps': camera.actual_fps,
            'running': camera.is_running()
        }

    def get_active_camera_ids(self):
        """Get list of active camera IDs."""
        return [cam_id for cam_id, cam in self.cameras.items() if cam.is_running()]

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
    """Test camera initialization and capture."""
    logger.info("Testing camera manager...")

    try:
        manager = CameraManager(num_cameras=1)
        manager.initialize_cameras([0])
        manager.start_all()

        # Capture a few frames
        for i in range(30):
            time.sleep(1 / 30)
            success, frame, frame_id, timestamp = manager.read_frame(0)
            if success:
                logger.info(f"Frame {frame_id}: {frame.shape}")
            else:
                logger.warning("Failed to read frame")

        manager.stop_all()
        logger.info("Camera test complete")

    except Exception as e:
        logger.error(f"Camera test failed: {e}")


if __name__ == '__main__':
    test_cameras()
