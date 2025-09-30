"""
Audio extraction module for the Video Summarizer project.
"""
import os
from pathlib import Path
from moviepy.editor import VideoFileClip
from typing import Optional, Union
from utils.logger import setup_logger
from config import AUDIO_SETTINGS, OUTPUT_DIR

logger = setup_logger(__name__)

class AudioExtractor:
    """Handles audio extraction from video files."""
    
    def __init__(self):
        self.supported_formats = AUDIO_SETTINGS["supported_formats"]
        self.output_dir = OUTPUT_DIR / "audio"
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_audio(self, video_path: Union[str, Path], 
                     output_name: Optional[str] = None) -> Path:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the input video file
            output_name: Optional custom name for output file
            
        Returns:
            Path to the extracted audio file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video format is not supported
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if video_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported video format: {video_path.suffix}")
        
        # Generate output filename
        if output_name is None:
            output_name = video_path.stem + ".wav"
        elif not output_name.endswith(".wav"):
            output_name += ".wav"
        
        output_path = self.output_dir / output_name
        
        try:
            logger.info(f"Extracting audio from {video_path}")
            
            # Load video and extract audio
            video = VideoFileClip(str(video_path))
            audio = video.audio
            
            if audio is None:
                raise ValueError("No audio track found in the video")
            
            # Write audio file
            audio.write_audiofile(
                str(output_path),
                verbose=False,
                logger=None
            )
            
            # Clean up
            audio.close()
            video.close()
            
            logger.info(f"Audio extracted successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise
    
    def get_audio_info(self, audio_path: Union[str, Path]) -> dict:
        """
        Get information about an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with audio file information
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            video = VideoFileClip(str(audio_path))
            audio = video.audio
            
            info = {
                "duration": audio.duration,
                "fps": audio.fps,
                "nchannels": audio.nchannels,
                "size": audio_path.stat().st_size
            }
            
            audio.close()
            video.close()
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting audio info: {str(e)}")
            raise

def main():
    """Example usage of the AudioExtractor class."""
    extractor = AudioExtractor()
    
    # Example: Extract audio from a video
    try:
        video_path = "video4.mp4"  # Replace with your video path
        audio_path = extractor.extract_audio(video_path)
        print(f"Audio extracted to: {audio_path}")
        
        # Get audio information
        info = extractor.get_audio_info(audio_path)
        print(f"Audio info: {info}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()