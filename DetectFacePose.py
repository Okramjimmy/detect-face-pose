"""
Enhanced Face Pose Detection System
Detects and classifies face poses as Frontal, Left Profile, or Right Profile
"""

from facenet_pytorch import MTCNN
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import requests
import argparse
import torch
import cv2
from pathlib import Path
from typing import Tuple, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FacePoseDetector:
    """Face pose detection using MTCNN and landmark analysis"""
    
    # Configuration constants
    FRONTAL_ANGLE_RANGE = (35, 58)
    DETECTION_THRESHOLD = 0.9
    FONT_SCALE = 2
    FONT_THICKNESS = 3
    LINE_THICKNESS = 3
    
    # Color definitions (BGR for OpenCV)
    COLORS = {
        'Frontal': (0, 255, 0),      # Green
        'Left Profile': (0, 0, 255),  # Red
        'Right Profile': (255, 0, 0)  # Blue
    }
    
    def __init__(self, device: str = 'auto'):
        """Initialize the face pose detector"""
        if device == 'auto':
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f'Running on device: {self.device}')
        
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=True  # Keep all detected faces
        )
    
    @staticmethod
    def calculate_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
        """
        Calculate angle at point_b formed by points a, b, c
        
        Args:
            point_a, point_b, point_c: 2D coordinates
            
        Returns:
            Angle in degrees
        """
        ba = point_a - point_b
        bc = point_c - point_b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Clip to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def classify_pose(self, angle_right: float, angle_left: float) -> str:
        """
        Classify face pose based on eye-nose angles
        
        Args:
            angle_right: Angle at right eye
            angle_left: Angle at left eye
            
        Returns:
            Pose classification: 'Frontal', 'Left Profile', or 'Right Profile'
        """
        min_angle, max_angle = self.FRONTAL_ANGLE_RANGE
        
        if (min_angle <= int(angle_right) <= max_angle and 
            min_angle <= int(angle_left) <= max_angle):
            return 'Frontal'
        elif angle_right < angle_left:
            return 'Left Profile'
        else:
            return 'Right Profile'
    
    def detect_faces(self, image: Image.Image) -> Tuple[Optional[np.ndarray], List[float], List[float], List[str]]:
        """
        Detect faces and classify their poses
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (landmarks, right_angles, left_angles, pose_labels)
        """
        try:
            bbox, prob, landmarks = self.mtcnn.detect(image, landmarks=True)
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None, [], [], []
        
        if bbox is None or landmarks is None:
            logger.warning("No faces detected in the image")
            return None, [], [], []
        
        angle_right_list = []
        angle_left_list = []
        pose_list = []
        valid_landmarks = []
        
        for i, (lm, confidence) in enumerate(zip(landmarks, prob)):
            if confidence < self.DETECTION_THRESHOLD:
                logger.info(f"Face {i+1} below confidence threshold: {confidence:.3f}")
                continue
            
            # Calculate angles using eye-nose landmarks
            # Landmarks: [Left Eye], [Right Eye], [Nose], [Left Mouth], [Right Mouth]
            angle_right = self.calculate_angle(lm[0], lm[1], lm[2])  # At right eye
            angle_left = self.calculate_angle(lm[1], lm[0], lm[2])   # At left eye
            
            pose = self.classify_pose(angle_right, angle_left)
            
            angle_right_list.append(angle_right)
            angle_left_list.append(angle_left)
            pose_list.append(pose)
            valid_landmarks.append(lm)
        
        if not valid_landmarks:
            logger.warning("No faces met the confidence threshold")
            return None, [], [], []
        
        return np.array(valid_landmarks), angle_right_list, angle_left_list, pose_list
    
    def visualize_static(self, image: Image.Image, landmarks: np.ndarray, 
                        angles_right: List[float], angles_left: List[float], 
                        poses: List[str], output_path: str = 'output_detection.jpg'):
        """Visualize detection results on static image"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Count poses
        pose_counts = {pose: poses.count(pose) for pose in set(poses)}
        total_faces = len(poses)
        
        title = f"Detected Faces: {total_faces}"
        if pose_counts:
            title += " | " + " | ".join([f"{pose}: {count}" for pose, count in pose_counts.items()])
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Display image
        ax.imshow(image)
        
        for lm, ang_r, ang_l, pose in zip(landmarks, angles_right, angles_left, poses):
            # Draw landmarks
            for point in lm:
                ax.scatter(point[0], point[1], c='cyan', s=50, edgecolors='black', linewidths=1.5)
            
            # Draw triangles connecting eyes and nose
            eye_nose_points = lm[:3]  # Left eye, right eye, nose
            triangle = plt.Polygon(eye_nose_points, fill=False, edgecolor='yellow', linewidth=2)
            ax.add_patch(triangle)
            
            # Add text annotation
            text_x = lm[2][0]  # Nose x-coordinate
            text_y = lm[2][1] - 30  # Above nose
            ax.text(text_x, text_y, 
                   f"{pose}\nL:{int(ang_l)}° R:{int(ang_r)}°",
                   size=12, ha="center", va="bottom",
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontweight='bold')
        
        ax.axis('off')
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved output to {output_path}")
    
    def visualize_video(self, frame: np.ndarray, landmarks: np.ndarray,
                       angles_right: List[float], angles_left: List[float],
                       poses: List[str]) -> np.ndarray:
        """Annotate video frame with detection results"""
        annotated_frame = frame.copy()
        
        for lm, ang_r, ang_l, pose in zip(landmarks, angles_right, angles_left, poses):
            color = self.COLORS.get(pose, (255, 255, 255))
            
            # Draw landmarks
            for point in lm:
                cv2.circle(annotated_frame, (int(point[0]), int(point[1])),
                          radius=5, color=(0, 255, 255), thickness=-1)
            
            # Draw lines connecting eyes and nose
            points = [(int(lm[i][0]), int(lm[i][1])) for i in range(3)]
            cv2.line(annotated_frame, points[0], points[1], (255, 255, 0), self.LINE_THICKNESS)
            cv2.line(annotated_frame, points[0], points[2], (255, 255, 0), self.LINE_THICKNESS)
            cv2.line(annotated_frame, points[1], points[2], (255, 255, 0), self.LINE_THICKNESS)
            
            # Add text label
            text = f"{pose} (L:{int(ang_l)} R:{int(ang_r)})"
            text_pos = (int(lm[2][0]) - 100, int(lm[2][1]) - 20)
            
            # Add background rectangle for better readability
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated_frame, 
                         (text_pos[0] - 5, text_pos[1] - text_height - 5),
                         (text_pos[0] + text_width + 5, text_pos[1] + 5),
                         (255, 255, 255), -1)
            
            cv2.putText(annotated_frame, text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        return annotated_frame


def process_image(detector: FacePoseDetector, image_source: str, is_url: bool = False):
    """Process a single image from path or URL"""
    try:
        if is_url:
            logger.info(f"Downloading image from URL: {image_source}")
            response = requests.get(image_source, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(response.raw)
        else:
            logger.info(f"Loading image from path: {image_source}")
            image = Image.open(image_source)
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert('RGB')
        
        # Detect faces
        landmarks, angles_r, angles_l, poses = detector.detect_faces(image)
        
        if landmarks is not None:
            detector.visualize_static(image, landmarks, angles_r, angles_l, poses)
            logger.info(f"Detected {len(poses)} face(s)")
        else:
            logger.warning("No faces detected or all faces below threshold")
            
    except Exception as e:
        logger.error(f"Error processing image: {e}")


def process_webcam(detector: FacePoseDetector, camera_id: int = 0):
    """Process video from webcam"""
    logger.info(f"Starting webcam capture (camera {camera_id})")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    window_name = 'Face Pose Detection - Press Q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 800)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            frame_count += 1
            
            # Process every frame (can skip frames for better performance)
            landmarks, angles_r, angles_l, poses = detector.detect_faces(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            
            if landmarks is not None:
                frame = detector.visualize_video(frame, landmarks, angles_r, angles_l, poses)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam capture ended")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Face Pose Detection - Detect and classify face orientations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process local image:    python script.py -p /path/to/image.jpg
  Process image from URL: python script.py -u https://example.com/image.jpg
  Use webcam:            python script.py
  Use specific camera:    python script.py -c 1
        """
    )
    
    parser.add_argument("-p", "--path", type=str, help="Path to input image")
    parser.add_argument("-u", "--url", type=str, help="URL of input image")
    parser.add_argument("-c", "--camera", type=int, default=0, 
                       help="Camera ID for webcam mode (default: 0)")
    parser.add_argument("-d", "--device", type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help="Device to run on (default: auto)")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FacePoseDetector(device=args.device)
    
    # Process based on input type
    if args.path:
        process_image(detector, args.path, is_url=False)
    elif args.url:
        process_image(detector, args.url, is_url=True)
    else:
        process_webcam(detector, args.camera)


if __name__ == '__main__':
    main()