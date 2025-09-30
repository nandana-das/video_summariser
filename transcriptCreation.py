"""
Transcription module for the Video Summarizer project.
"""
import os
import json
import torchaudio
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
from vosk import Model, KaldiRecognizer
import zipfile
import urllib.request
from utils.logger import setup_logger
from config import TRANSCRIPTION_SETTINGS, MODELS_DIR, OUTPUT_DIR

logger = setup_logger(__name__)

class TranscriptGenerator:
    """Handles audio transcription using Vosk speech recognition."""
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the transcript generator.
        
        Args:
            model_path: Path to the Vosk model directory
        """
        self.model_path = Path(model_path) if model_path else MODELS_DIR / TRANSCRIPTION_SETTINGS["model_name"]
        self.model = None
        self.output_dir = OUTPUT_DIR / "transcripts"
        self.output_dir.mkdir(exist_ok=True)
        
        # Download model if not present
        self._ensure_model_available()
        
        # Load the model
        self._load_model()
    
    def _ensure_model_available(self):
        """Download and extract the Vosk model if not present."""
        if not self.model_path.exists():
            logger.info(f"Model not found at {self.model_path}. Downloading...")
            self._download_model()
    
    def _download_model(self):
        """Download and extract the Vosk model."""
        try:
            model_url = TRANSCRIPTION_SETTINGS["model_url"]
            zip_path = self.model_path.parent / f"{self.model_path.name}.zip"
            
            logger.info(f"Downloading model from {model_url}")
            urllib.request.urlretrieve(model_url, zip_path)
            
            logger.info("Extracting model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.model_path.parent)
            
            # Clean up zip file
            zip_path.unlink()
            
            logger.info("Model downloaded and extracted successfully")
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise
    
    def _load_model(self):
        """Load the Vosk model."""
        try:
            logger.info(f"Loading Vosk model from {self.model_path}")
            self.model = Model(str(self.model_path))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_path: Union[str, Path], 
                        output_name: Optional[str] = None,
                        confidence_threshold: Optional[float] = None) -> Path:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            output_name: Optional custom name for output file
            confidence_threshold: Minimum confidence threshold for transcription
            
        Returns:
            Path to the transcript file
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If model is not loaded
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Generate output filename
        if output_name is None:
            output_name = audio_path.stem + "_transcript.txt"
        elif not output_name.endswith(".txt"):
            output_name += ".txt"
        
        output_path = self.output_dir / output_name
        
        try:
            logger.info(f"Transcribing audio: {audio_path}")

            # Load audio file
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Convert to 16-bit PCM
            audio_data = waveform.numpy()
            audio_data = (audio_data * 32767).astype(np.int16)

            # Initialize recognizer
            rec = KaldiRecognizer(self.model, sample_rate)
            
            # Process audio in chunks
            transcript_parts = []
            chunk_size = 500
            
            for i in range(0, len(audio_data), chunk_size):
                data = audio_data[i:i+chunk_size].tobytes()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        transcript_parts.append(result["text"])
            
            # Finalize transcription
            result = json.loads(rec.FinalResult())
            if result.get("text"):
                transcript_parts.append(result["text"])
            
            # Combine all parts
            transcript = " ".join(transcript_parts)
            
            # Save transcript
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            logger.info(f"Transcript saved successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise
    
    def get_transcription_info(self, transcript_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a transcript file.
        
        Args:
            transcript_path: Path to the transcript file
            
        Returns:
            Dictionary with transcript information
        """
        transcript_path = Path(transcript_path)
        
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            info = {
                "file_size": transcript_path.stat().st_size,
                "word_count": len(content.split()),
                "character_count": len(content),
                "line_count": len(content.splitlines())
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting transcript info: {str(e)}")
            raise

def main():
    """Example usage of the TranscriptGenerator class."""
    generator = TranscriptGenerator()
    
    # Example: Transcribe an audio file
    try:
        audio_path = "output/audio/audio4.wav"  # Replace with your audio path
        transcript_path = generator.transcribe_audio(audio_path)
        print(f"Transcript saved to: {transcript_path}")
        
        # Get transcript information
        info = generator.get_transcription_info(transcript_path)
        print(f"Transcript info: {info}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()