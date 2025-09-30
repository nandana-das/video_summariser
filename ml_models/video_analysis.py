"""
Advanced computer vision module for video analysis, scene detection, and visual understanding.
"""
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import moviepy.editor as mp
from transformers import (
    VideoMAEForVideoClassification, VideoMAEImageProcessor,
    AutoImageProcessor, AutoModelForImageClassification,
    BlipProcessor, BlipForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
import mlflow
import mlflow.pytorch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from config import OUTPUT_DIR

logger = setup_logger(__name__)

class VideoAnalyzer:
    """Advanced video analysis using computer vision and deep learning."""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the video analyzer.
        
        Args:
            device: Device to run inference on
        """
        self.device = self._get_device(device)
        self.scene_detector = None
        self.object_detector = None
        self.action_classifier = None
        self.caption_generator = None
        self.similarity_model = None
        
        # Initialize models
        self._load_models()
        
        logger.info(f"VideoAnalyzer initialized on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_models(self):
        """Load computer vision models."""
        try:
            # Load YOLO for object detection
            try:
                from ultralytics import YOLO
                self.object_detector = YOLO('yolov8n.pt')
                logger.info("Loaded YOLO object detector")
            except ImportError:
                logger.warning("YOLO not available, using alternative object detection")
                self.object_detector = None
        except Exception as e:
            logger.warning(f"Error loading YOLO: {e}")
            self.object_detector = None
            
            # Load BLIP for image captioning
            try:
                self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.caption_generator = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to(self.device)
                logger.info("Loaded BLIP caption generator")
            except Exception as e:
                logger.warning(f"Failed to load BLIP: {str(e)}")
                self.caption_generator = None
            
            # Load sentence transformer for similarity
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def extract_frames(self, video_path: Union[str, Path], 
                      fps: float = 1.0, max_frames: int = 100) -> List[np.ndarray]:
        """
        Extract frames from video at specified FPS.
        
        Args:
            video_path: Path to video file
            fps: Frames per second to extract
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frames
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Load video
            cap = cv2.VideoCapture(str(video_path))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval
            frame_interval = max(1, int(video_fps / fps))
            
            frames = []
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
    
    def detect_scenes(self, video_path: Union[str, Path], 
                     threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detect scene changes in video.
        
        Args:
            video_path: Path to video file
            threshold: Threshold for scene change detection
            
        Returns:
            List of scene information
        """
        try:
            # Extract frames
            frames = self.extract_frames(video_path, fps=2.0)
            
            if len(frames) < 2:
                return []
            
            scenes = []
            current_scene_start = 0
            
            for i in range(1, len(frames)):
                # Calculate frame difference
                frame_diff = self._calculate_frame_difference(frames[i-1], frames[i])
                
                if frame_diff > threshold:
                    # Scene change detected
                    scene_info = {
                        'start_frame': current_scene_start,
                        'end_frame': i - 1,
                        'start_time': current_scene_start / 2.0,  # Assuming 2 FPS
                        'end_time': (i - 1) / 2.0,
                        'duration': (i - current_scene_start) / 2.0,
                        'change_score': frame_diff
                    }
                    scenes.append(scene_info)
                    current_scene_start = i
            
            # Add final scene
            if current_scene_start < len(frames) - 1:
                scene_info = {
                    'start_frame': current_scene_start,
                    'end_frame': len(frames) - 1,
                    'start_time': current_scene_start / 2.0,
                    'end_time': (len(frames) - 1) / 2.0,
                    'duration': (len(frames) - current_scene_start) / 2.0,
                    'change_score': 0.0
                }
                scenes.append(scene_info)
            
            logger.info(f"Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            logger.error(f"Error detecting scenes: {str(e)}")
            raise
    
    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate difference between two frames."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Calculate structural similarity
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(gray1, gray2)
        
        # Return difference (1 - similarity)
        return 1 - ssim
    
    def detect_objects(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect objects in video frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of object detection results
        """
        try:
            if self.object_detector is None:
                logger.warning("Object detector not available")
                return []
            
            all_detections = []
            
            for i, frame in enumerate(frames):
                # Run object detection
                results = self.object_detector(frame)
                
                frame_detections = {
                    'frame_index': i,
                    'objects': []
                }
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            obj_info = {
                                'class': self.object_detector.names[int(box.cls)],
                                'confidence': float(box.conf),
                                'bbox': box.xyxy[0].tolist(),
                                'center': [(box.xyxy[0][0] + box.xyxy[0][2]) / 2,
                                         (box.xyxy[0][1] + box.xyxy[0][3]) / 2]
                            }
                            frame_detections['objects'].append(obj_info)
                
                all_detections.append(frame_detections)
            
            logger.info(f"Detected objects in {len(frames)} frames")
            return all_detections
            
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            raise
    
    def generate_captions(self, frames: List[np.ndarray]) -> List[str]:
        """
        Generate captions for video frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of generated captions
        """
        try:
            if self.caption_generator is None:
                logger.warning("Caption generator not available")
                return []
            
            captions = []
            
            for frame in frames:
                # Convert numpy array to PIL Image
                image = Image.fromarray(frame)
                
                # Generate caption
                inputs = self.caption_processor(image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    out = self.caption_generator.generate(**inputs, max_length=50)
                
                caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)
            
            logger.info(f"Generated {len(captions)} captions")
            return captions
            
        except Exception as e:
            logger.error(f"Error generating captions: {str(e)}")
            raise
    
    def analyze_video_content(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Comprehensive video content analysis.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Comprehensive analysis results
        """
        try:
            logger.info(f"Analyzing video: {video_path}")
            
            # Extract frames
            frames = self.extract_frames(video_path, fps=1.0, max_frames=50)
            
            # Detect scenes
            scenes = self.detect_scenes(video_path)
            
            # Detect objects
            object_detections = self.detect_objects(frames)
            
            # Generate captions
            captions = self.generate_captions(frames)
            
            # Analyze visual content
            visual_features = self._extract_visual_features(frames)
            
            # Generate summary
            content_summary = self._generate_content_summary(
                scenes, object_detections, captions, visual_features
            )
            
            result = {
                'video_path': str(video_path),
                'total_frames': len(frames),
                'scenes': scenes,
                'object_detections': object_detections,
                'captions': captions,
                'visual_features': visual_features,
                'content_summary': content_summary,
                'analysis_metadata': {
                    'frames_analyzed': len(frames),
                    'scenes_detected': len(scenes),
                    'objects_detected': sum(len(d['objects']) for d in object_detections),
                    'captions_generated': len(captions)
                }
            }
            
            logger.info("Video analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            raise
    
    def _extract_visual_features(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Extract visual features from frames."""
        try:
            features = {
                'brightness': [],
                'contrast': [],
                'color_diversity': [],
                'motion_vectors': []
            }
            
            for frame in frames:
                # Brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray)
                features['brightness'].append(brightness)
                
                # Contrast
                contrast = np.std(gray)
                features['contrast'].append(contrast)
                
                # Color diversity (number of unique colors)
                unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
                features['color_diversity'].append(unique_colors)
            
            # Calculate motion vectors between consecutive frames
            for i in range(1, len(frames)):
                motion = self._calculate_motion(frames[i-1], frames[i])
                features['motion_vectors'].append(motion)
            
            # Calculate statistics
            for key in features:
                if features[key]:
                    features[f'{key}_mean'] = np.mean(features[key])
                    features[f'{key}_std'] = np.std(features[key])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {str(e)}")
            return {}
    
    def _calculate_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate motion between two frames."""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                gray1, gray2, 
                np.array([[100, 100]], dtype=np.float32),
                None
            )[0]
            
            if flow is not None and len(flow) > 0:
                motion = np.linalg.norm(flow[0])
            else:
                motion = 0.0
            
            return motion
            
        except Exception as e:
            logger.warning(f"Error calculating motion: {str(e)}")
            return 0.0
    
    def _generate_content_summary(self, scenes: List[Dict], 
                                object_detections: List[Dict],
                                captions: List[str],
                                visual_features: Dict) -> str:
        """Generate a summary of video content."""
        try:
            summary_parts = []
            
            # Scene information
            if scenes:
                summary_parts.append(f"Video contains {len(scenes)} scenes with an average duration of {np.mean([s['duration'] for s in scenes]):.1f} seconds.")
            
            # Object information
            all_objects = []
            for detection in object_detections:
                all_objects.extend([obj['class'] for obj in detection['objects']])
            
            if all_objects:
                from collections import Counter
                object_counts = Counter(all_objects)
                top_objects = object_counts.most_common(5)
                summary_parts.append(f"Main objects detected: {', '.join([f'{obj} ({count})' for obj, count in top_objects])}.")
            
            # Caption information
            if captions:
                # Use sentence transformer to find most representative captions
                if self.similarity_model and len(captions) > 1:
                    embeddings = self.similarity_model.encode(captions)
                    # Find caption most similar to all others
                    similarities = np.dot(embeddings, embeddings.T)
                    representative_idx = np.argmax(np.sum(similarities, axis=1))
                    summary_parts.append(f"Key visual content: {captions[representative_idx]}")
            
            # Visual characteristics
            if visual_features:
                avg_brightness = visual_features.get('brightness_mean', 0)
                avg_contrast = visual_features.get('contrast_mean', 0)
                summary_parts.append(f"Visual characteristics: average brightness {avg_brightness:.1f}, contrast {avg_contrast:.1f}.")
            
            return " ".join(summary_parts) if summary_parts else "No significant content detected."
            
        except Exception as e:
            logger.error(f"Error generating content summary: {str(e)}")
            return "Error generating content summary."
    
    def save_analysis(self, analysis: Dict[str, Any], 
                     output_path: Union[str, Path]):
        """Save video analysis results."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=== VIDEO ANALYSIS REPORT ===\n\n")
                f.write(f"Video: {analysis['video_path']}\n")
                f.write(f"Frames analyzed: {analysis['analysis_metadata']['frames_analyzed']}\n")
                f.write(f"Scenes detected: {analysis['analysis_metadata']['scenes_detected']}\n")
                f.write(f"Objects detected: {analysis['analysis_metadata']['objects_detected']}\n\n")
                
                f.write("=== CONTENT SUMMARY ===\n")
                f.write(analysis['content_summary'] + "\n\n")
                
                f.write("=== SCENES ===\n")
                for i, scene in enumerate(analysis['scenes']):
                    f.write(f"Scene {i+1}: {scene['start_time']:.1f}s - {scene['end_time']:.1f}s "
                           f"(duration: {scene['duration']:.1f}s)\n")
                
                f.write("\n=== OBJECT DETECTIONS ===\n")
                for i, detection in enumerate(analysis['object_detections']):
                    if detection['objects']:
                        f.write(f"Frame {i}: {', '.join([obj['class'] for obj in detection['objects']])}\n")
            
            logger.info(f"Analysis saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            raise

def main():
    """Example usage of the VideoAnalyzer."""
    analyzer = VideoAnalyzer()
    
    # Example video file (replace with actual path)
    video_path = "example_video.mp4"
    
    if Path(video_path).exists():
        # Analyze video
        analysis = analyzer.analyze_video_content(video_path)
        
        print("=== VIDEO ANALYSIS ===")
        print(f"Scenes: {len(analysis['scenes'])}")
        print(f"Objects detected: {analysis['analysis_metadata']['objects_detected']}")
        print(f"Content summary: {analysis['content_summary']}")
    else:
        print(f"Video file not found: {video_path}")

if __name__ == "__main__":
    main()
