#!/usr/bin/env python3
"""
Camera Manager for multiple ArduCam handling on Raspberry Pi 5
Handles frame capture from multiple cameras with threading support
"""

import cv2
import threading
import time
import logging
from collections import deque
from config import cfg

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraStream:
    """Thread-safe camera stream handler for a single camera"""

    def __init__(self, camera_id: int, width: int = None, height: int = None, fps: int = None):
        """
        Initialize camera stream

        Args:
            camera_id: Camera device index
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.camera_id = camera_id
        self.width = width or cfg.camera.frame_width
        self.height = height or cfg.camera.frame_height
        self.fps = fps or cfg.camera.fps

        self.cap = None
        self.frame = None
        self.frame_id = 0
        self.timestamp = 0
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize the camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read actual properties
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Camera {self.camera_id} initialized: {self.actual_width}x{self.actual_height} @ {self.actual_fps}fps")

    def start(self):
        """Start the camera capture thread"""
        if self.running:
            return self

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        """Continuous frame capture loop"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                    self.timestamp = time.time()
            else:
                logger.warning(f"Camera {self.camera_id}: Failed to read frame")
                time.sleep(0.01)

    def read(self):
        """
        Read the latest frame

        Returns:
            tuple: (success, frame, frame_id, timestamp)
        """
        with self.lock:
            if self.frame is None:
                return False, None, 0, 0
            return True, self.frame.copy(), self.frame_id, self.timestamp

    def stop(self):
        """Stop the camera capture"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        logger.info(f"Camera {self.camera_id} stopped")

    def is_running(self):
        """Check if camera is running"""
        return self.running and self.cap is not None and self.cap.isOpened()


class CameraManager:
    """Manager for multiple camera streams"""

    def __init__(self, num_cameras: int = None):
        """
        Initialize camera manager

        Args:
            num_cameras: Number of cameras to manage
        """
        self.num_cameras = num_cameras or cfg.camera.num_cameras
        self.cameras = {}
        self.running = False

    def initialize_cameras(self, camera_ids: list = None):
        """
        Initialize all cameras

        Args:
            camera_ids: List of camera device IDs. Defaults to [0, 1, 2, 3] for 4 cameras
        """
        if camera_ids is None:
            camera_ids = list(range(self.num_cameras))

        for cam_id in camera_ids:
            try:
                camera = CameraStream(cam_id)
                self.cameras[cam_id] = camera
                logger.info(f"Initialized camera {cam_id}")
            except RuntimeError as e:
                logger.error(f"Failed to initialize camera {cam_id}: {e}")

        if not self.cameras:
            raise RuntimeError("No cameras could be initialized")

        logger.info(f"Initialized {len(self.cameras)} cameras")

    def start_all(self):
        """Start all camera streams"""
        self.running = True
        for cam_id, camera in self.cameras.items():
            camera.start()
            logger.info(f"Started camera {cam_id}")

    def stop_all(self):
        """Stop all camera streams"""
        self.running = False
        for cam_id, camera in self.cameras.items():
            camera.stop()
        logger.info("All cameras stopped")

    def read_frame(self, camera_id: int):
        """
        Read frame from a specific camera

        Args:
            camera_id: Camera ID to read from

        Returns:
            tuple: (success, frame, frame_id, timestamp)
        """
        if camera_id not in self.cameras:
            return False, None, 0, 0
        return self.cameras[camera_id].read()

    def read_all_frames(self):
        """
        Read frames from all cameras

        Returns:
            dict: {camera_id: (success, frame, frame_id, timestamp)}
        """
        frames = {}
        for cam_id, camera in self.cameras.items():
            frames[cam_id] = camera.read()
        return frames

    def get_camera_info(self, camera_id: int):
        """
        Get camera information

        Args:
            camera_id: Camera ID

        Returns:
            dict: Camera properties
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
        """Get list of active camera IDs"""
        return [cam_id for cam_id, cam in self.cameras.items() if cam.is_running()]

    def __enter__(self):
        """Context manager entry"""
        self.initialize_cameras()
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_all()
        return False


def test_cameras():
    """Test camera initialization and capture"""
    logger.info("Testing camera manager...")

    try:
        manager = CameraManager(num_cameras=1)  # Test with 1 camera first
        manager.initialize_cameras([0])
        manager.start_all()

        # Capture a few frames
        for i in range(30):
            time.sleep(1/30)
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
