#!/usr/bin/env python3
"""
Wildlife Camera Detection and Recording System
Main entry point for Raspberry Pi 5 with multiple ArduCams

This system:
1. Captures frames from multiple ArduCams
2. Runs animal detection on each camera feed
3. Tracks detected animals across frames
4. Records video when animals are confirmed (not false positives)
5. Spawns individual videos for each tracked animal
6. Uploads recordings to S3 in background threads
"""

import sys
import time
import signal
import logging
import argparse
import threading
from typing import Dict, Optional
import cv2
import numpy as np

from config import cfg
from inference import Inference
from tracker import Tracker
from camera_manager import CameraManager
from recording_manager import RecordingManager

# Conditional S3 import
try:
    from s3_manager import S3Manager
    S3_AVAILABLE = True
except Exception as e:
    S3_AVAILABLE = False
    logging.warning(f"S3 manager not available: {e}")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class CameraPipeline:
    """Processing pipeline for a single camera"""

    def __init__(self, camera_id: int, detector: Inference, s3_manager=None):
        """
        Initialize camera pipeline

        Args:
            camera_id: Camera identifier
            detector: Shared Inference instance
            s3_manager: Optional S3Manager for uploads
        """
        self.camera_id = camera_id
        self.detector = detector
        self.tracker = Tracker(cfg)
        self.recording_manager = RecordingManager(camera_id, s3_manager)

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.fps_frame_count = 0

        logger.info(f"Pipeline initialized for camera {camera_id}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through detection, tracking, and recording

        Args:
            frame: Input video frame

        Returns:
            Annotated frame
        """
        self.frame_count += 1
        self.fps_frame_count += 1

        # Calculate FPS every second
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.fps_frame_count / (current_time - self.last_fps_time)
            self.fps_frame_count = 0
            self.last_fps_time = current_time

        # Run detection
        annotated_frame, boxes, confidences, class_ids = self.detector.infer(frame)

        # Run tracking
        tracked_objects = self.tracker.update(frame, boxes, confidences, class_ids)

        # Process recording
        annotated_frame = self.recording_manager.process_frame(annotated_frame, tracked_objects)

        # Add FPS and camera info overlay
        if cfg.flags.render_fps:
            info_text = f"Cam{self.camera_id} | FPS: {self.fps:.1f} | Frame: {self.frame_count}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add recording status
            status = self.recording_manager.get_status()
            if status['is_recording']:
                rec_text = f"REC | Tracks: {status['recording_candidates']}"
                cv2.putText(annotated_frame, rec_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated_frame

    def get_status(self) -> dict:
        """Get pipeline status"""
        return {
            'camera_id': self.camera_id,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'recording': self.recording_manager.get_status()
        }

    def stop(self):
        """Stop the pipeline"""
        self.recording_manager.stop()
        logger.info(f"Pipeline stopped for camera {self.camera_id}")


class WildlifeMonitoringSystem:
    """Main wildlife monitoring system coordinating all cameras"""

    def __init__(self, camera_ids: list = None, use_s3: bool = True):
        """
        Initialize the wildlife monitoring system

        Args:
            camera_ids: List of camera device IDs to use
            use_s3: Whether to enable S3 uploads
        """
        self.running = False
        self.camera_ids = camera_ids or list(range(cfg.camera.num_cameras))

        # Initialize S3 manager if enabled
        self.s3_manager = None
        if use_s3 and cfg.s3.enable and S3_AVAILABLE:
            try:
                self.s3_manager = S3Manager()
                logger.info("S3 manager initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 manager: {e}")

        # Initialize shared detector (single instance for efficiency)
        logger.info("Loading detection model...")
        self.detector = Inference()
        logger.info("Detection model loaded")

        # Initialize camera manager
        self.camera_manager = CameraManager(num_cameras=len(self.camera_ids))

        # Initialize processing pipelines for each camera
        self.pipelines: Dict[int, CameraPipeline] = {}

        # Status tracking
        self.start_time = None
        self.total_frames_processed = 0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Wildlife Monitoring System initialized with cameras: {self.camera_ids}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received...")
        self.stop()

    def _initialize_pipelines(self):
        """Initialize processing pipelines for all cameras"""
        for cam_id in self.camera_manager.get_active_camera_ids():
            self.pipelines[cam_id] = CameraPipeline(cam_id, self.detector, self.s3_manager)

    def start(self):
        """Start the monitoring system"""
        logger.info("Starting Wildlife Monitoring System...")

        try:
            # Initialize cameras
            self.camera_manager.initialize_cameras(self.camera_ids)
            self.camera_manager.start_all()

            # Initialize pipelines
            self._initialize_pipelines()

            self.running = True
            self.start_time = time.time()

            logger.info("System started, beginning main loop...")
            self._main_loop()

        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.stop()
            raise

    def _main_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Read frames from all cameras
                frames = self.camera_manager.read_all_frames()

                for cam_id, (success, frame, frame_id, timestamp) in frames.items():
                    if not success or frame is None:
                        continue

                    # Process frame through pipeline
                    if cam_id in self.pipelines:
                        processed_frame = self.pipelines[cam_id].process_frame(frame)
                        self.total_frames_processed += 1

                        # Optional: display frames (disabled for headless operation)
                        if cfg.flags.image_show:
                            cv2.imshow(f"Camera {cam_id}", processed_frame)

                # Handle display events if showing
                if cfg.flags.image_show:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit key pressed")
                        self.running = False
                    elif key == ord('s'):
                        # Print status on 's' key
                        self._print_status()

                # Small delay to prevent CPU overload
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(0.1)

    def _print_status(self):
        """Print system status"""
        runtime = time.time() - self.start_time if self.start_time else 0
        logger.info("=" * 50)
        logger.info(f"System Runtime: {runtime:.1f}s")
        logger.info(f"Total Frames Processed: {self.total_frames_processed}")

        for cam_id, pipeline in self.pipelines.items():
            status = pipeline.get_status()
            logger.info(f"Camera {cam_id}: FPS={status['fps']:.1f}, "
                        f"Frames={status['frame_count']}, "
                        f"Recording={status['recording']['is_recording']}")
        logger.info("=" * 50)

    def stop(self):
        """Stop the monitoring system"""
        logger.info("Stopping Wildlife Monitoring System...")
        self.running = False

        # Stop all pipelines
        for pipeline in self.pipelines.values():
            pipeline.stop()

        # Stop cameras
        self.camera_manager.stop_all()

        # Close any display windows
        cv2.destroyAllWindows()

        # Print final status
        if self.start_time:
            runtime = time.time() - self.start_time
            logger.info(f"System stopped. Total runtime: {runtime:.1f}s, "
                        f"Total frames: {self.total_frames_processed}")

    def get_status(self) -> dict:
        """Get complete system status"""
        return {
            'running': self.running,
            'runtime': time.time() - self.start_time if self.start_time else 0,
            'total_frames': self.total_frames_processed,
            'cameras': {cam_id: pipeline.get_status() for cam_id, pipeline in self.pipelines.items()}
        }


def run_single_camera_test(camera_id: int = 0):
    """Run a test with a single camera (for development/debugging)"""
    logger.info(f"Running single camera test on camera {camera_id}")

    # Initialize components
    detector = Inference()
    tracker = Tracker(cfg)

    # Try to initialize S3
    s3_manager = None
    if cfg.s3.enable and S3_AVAILABLE:
        try:
            s3_manager = S3Manager()
        except Exception as e:
            logger.warning(f"S3 not available: {e}")

    recording_manager = RecordingManager(camera_id, s3_manager)

    # Open camera via CameraManager (uses Picamera2 for Pi CSI cameras)
    cam_manager = CameraManager(num_cameras=1)
    try:
        cam_manager.initialize_cameras([camera_id])
        cam_manager.start_all()
    except RuntimeError as e:
        logger.error(f"Failed to open camera {camera_id}: {e}")
        return

    logger.info("Starting single camera test loop (press Ctrl+C to quit)...")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame, _, _ = cam_manager.read_frame(camera_id)
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1

            # Detection
            annotated_frame, boxes, confidences, class_ids = detector.infer(frame)

            # Tracking
            tracked_objects = tracker.update(frame, boxes, confidences, class_ids)

            # Recording
            annotated_frame = recording_manager.process_frame(annotated_frame, tracked_objects)

            # Add info overlay
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Frame: {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display if show mode is on
            if cfg.flags.image_show:
                cv2.imshow(f"Camera {camera_id} Test", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        logger.info("Test interrupted")
    finally:
        cam_manager.stop_all()
        recording_manager.stop()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    logger.info(f"Test complete. Processed {frame_count} frames in {elapsed:.1f}s "
                f"(avg {frame_count/elapsed:.1f} fps)")


def run_video_test(video_path: str):
    """Run test on a video file instead of camera"""
    logger.info(f"Running video test on: {video_path}")

    # Initialize components
    detector = Inference()
    tracker = Tracker(cfg)
    recording_manager = RecordingManager(camera_id=0, s3_manager=None)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {total_frames} frames @ {fps} fps")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detection
            annotated_frame, boxes, confidences, class_ids = detector.infer(frame)

            # Tracking
            tracked_objects = tracker.update(frame, boxes, confidences, class_ids)

            # Recording
            annotated_frame = recording_manager.process_frame(annotated_frame, tracked_objects)

            # Progress
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames "
                            f"({100*frame_count/total_frames:.1f}%)")

            # Display (optional)
            if cfg.flags.image_show:
                cv2.imshow("Video Test", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        logger.info("Test interrupted")
    finally:
        cap.release()
        recording_manager.stop()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    logger.info(f"Test complete. Processed {frame_count} frames in {elapsed:.1f}s "
                f"(avg {frame_count/elapsed:.1f} fps)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Wildlife Camera Monitoring System")
    parser.add_argument('--cameras', type=int, nargs='+', default=None,
                        help='Libcamera device indices to use (default: 0 1 2 3 for ArduCam Quad Kit). '
                             'Check available cameras with: libcamera-hello --list-cameras')
    parser.add_argument('--single-camera', type=int, default=None,
                        help='Run single camera test mode with specified camera ID')
    parser.add_argument('--video', type=str, default=None,
                        help='Run on video file instead of camera')
    parser.add_argument('--no-s3', action='store_true',
                        help='Disable S3 uploads')
    parser.add_argument('--show', action='store_true',
                        help='Show video preview windows')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Configure display
    if args.show:
        cfg.flags.image_show = True

    # Run appropriate mode
    if args.video:
        run_video_test(args.video)
    elif args.single_camera is not None:
        run_single_camera_test(args.single_camera)
    else:
        camera_ids = args.cameras or list(range(cfg.camera.num_cameras))
        system = WildlifeMonitoringSystem(
            camera_ids=camera_ids,
            use_s3=not args.no_s3
        )
        try:
            system.start()
        except KeyboardInterrupt:
            pass
        finally:
            system.stop()


if __name__ == '__main__':
    main()
