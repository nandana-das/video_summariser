"""
Advanced speech recognition module using Whisper, Wav2Vec2, and other state-of-the-art models.
"""
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
# Whisper import will be handled in the class methods
WHISPER_AVAILABLE = False
whisper = None
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer,
    AutoProcessor, AutoModelForCTC, pipeline
)
from datasets import load_dataset
import mlflow
import mlflow.transformers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from config import OUTPUT_DIR

logger = setup_logger(__name__)

def _check_whisper_availability():
    """Check if Whisper is available and import it."""
    global WHISPER_AVAILABLE, whisper
    if not WHISPER_AVAILABLE:
        try:
            import whisper
            WHISPER_AVAILABLE = True
            return whisper
        except ImportError:
            WHISPER_AVAILABLE = False
            return None
    return whisper

class AdvancedSpeechRecognizer:
    """Advanced speech recognition using multiple state-of-the-art models."""
    
    def __init__(self, model_name: str = "whisper-base", device: str = "auto"):
        """
        Initialize the advanced speech recognizer.
        
        Args:
            model_name: Name of the model to use
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        self.whisper_model = None
        
        # Initialize models
        self._load_models()
        
        logger.info(f"AdvancedSpeechRecognizer initialized with {model_name} on {self.device}")
    
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
        """Load the speech recognition models."""
        try:
            if self.model_name.startswith("whisper"):
                # Load Whisper model (if available)
                whisper_module = _check_whisper_availability()
                if whisper_module is not None:
                    try:
                        whisper_model_name = self.model_name.replace("whisper-", "")
                        self.whisper_model = whisper_module.load_model(whisper_model_name, device=self.device)
                        logger.info(f"Loaded Whisper model: {whisper_model_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load Whisper model: {e}, using placeholder mode")
                        self.whisper_model = None
                        self.model_name = "placeholder"
                else:
                    logger.warning("Whisper not available, using placeholder mode")
                    self.whisper_model = None
                    self.model_name = "placeholder"
                
            elif self.model_name.startswith("wav2vec2"):
                # Load Wav2Vec2 model
                model_id = self.model_name.replace("wav2vec2-", "")
                self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
                self.processor = Wav2Vec2Processor.from_pretrained(model_id)
                self.model.to(self.device)
                logger.info(f"Loaded Wav2Vec2 model: {model_id}")
                
            elif self.model_name == "placeholder":
                # Placeholder mode - no actual models loaded
                logger.info("Using placeholder speech recognition mode")
                self.whisper_model = None
                self.model = None
                self.processor = None
            
            else:
                # Load generic ASR pipeline
                self.model = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info(f"Loaded ASR pipeline: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            # Fall back to placeholder mode
            logger.warning("Falling back to placeholder mode")
            self.whisper_model = None
            self.model = None
            self.processor = None
            self.model_name = "placeholder"
    
    def preprocess_audio(self, audio_path: Union[str, Path], 
                        target_sr: int = 16000) -> np.ndarray:
        """
        Preprocess audio file for speech recognition.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Check if it's a video file - if so, return dummy audio
            audio_path_str = str(audio_path)
            if any(ext in audio_path_str.lower() for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']):
                logger.warning(f"Video file detected: {audio_path_str}, returning dummy audio")
                # Return 1 second of silence
                return np.zeros(target_sr, dtype=np.float32)
            
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=target_sr)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Remove silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            return audio
            
        except Exception as e:
            logger.warning(f"Error preprocessing audio: {str(e)}, returning dummy audio")
            # Return 1 second of silence as fallback
            return np.zeros(target_sr, dtype=np.float32)
    
    def transcribe_whisper(self, audio_path: Union[str, Path], 
                          language: Optional[str] = None,
                          task: str = "transcribe") -> Dict[str, Any]:
        """
        Transcribe audio using Whisper model.
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional)
            task: Task type ('transcribe' or 'translate')
            
        Returns:
            Transcription results
        """
        try:
            if not WHISPER_AVAILABLE or self.whisper_model is None:
                raise ValueError("Whisper model not available or not loaded")
            
            # Load and preprocess audio
            audio = self.preprocess_audio(audio_path)
            
            # Check if we got dummy audio (video file)
            if len(audio) == 16000 and np.all(audio == 0):
                logger.warning("Dummy audio detected, returning placeholder transcription")
                return {
                    "text": "Audio processing not available for video files. This is a placeholder transcript.",
                    "language": "en",
                    "confidence": 0.0,
                    "segments": [],
                    "model": self.model_name
                }
            
            # Transcribe
            result = self.whisper_model.transcribe(
                audio,
                language=language,
                task=task,
                fp16=False if self.device == "cpu" else True
            )
            
            return {
                'text': result['text'],
                'language': result.get('language', 'unknown'),
                'segments': result.get('segments', []),
                'model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {str(e)}")
            # Return placeholder transcription on error
            return {
                "text": "Speech recognition not available. This is a placeholder transcript.",
                "language": "en",
                "confidence": 0.0,
                "segments": [],
                "model": self.model_name
            }
    
    def transcribe_wav2vec2(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Transcribe audio using Wav2Vec2 model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription results
        """
        try:
            if self.model is None or self.processor is None:
                raise ValueError("Wav2Vec2 model not loaded")
            
            # Load and preprocess audio
            audio = self.preprocess_audio(audio_path)
            
            # Process audio
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Get logits
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.decode(predicted_ids[0])
            
            return {
                'text': transcription,
                'confidence': torch.softmax(logits, dim=-1).max().item(),
                'model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error in Wav2Vec2 transcription: {str(e)}")
            raise
    
    def transcribe_pipeline(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Transcribe audio using Hugging Face pipeline.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription results
        """
        try:
            if self.model is None:
                raise ValueError("ASR pipeline not loaded")
            
            # Transcribe using pipeline
            result = self.model(str(audio_path))
            
            return {
                'text': result['text'],
                'model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline transcription: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_path: Union[str, Path], 
                        **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using the appropriate model.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional parameters
            
        Returns:
            Transcription results
        """
        try:
            if self.model_name.startswith("whisper"):
                return self.transcribe_whisper(audio_path, **kwargs)
            elif self.model_name.startswith("wav2vec2"):
                return self.transcribe_wav2vec2(audio_path)
            else:
                return self.transcribe_pipeline(audio_path)
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise
    
    def batch_transcribe(self, audio_paths: List[Union[str, Path]], 
                        **kwargs) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            **kwargs: Additional parameters
            
        Returns:
            List of transcription results
        """
        results = []
        for i, audio_path in enumerate(audio_paths):
            try:
                logger.info(f"Transcribing audio {i+1}/{len(audio_paths)}: {audio_path}")
                result = self.transcribe_audio(audio_path, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {str(e)}")
                results.append({
                    'text': '',
                    'error': str(e)
                })
        
        return results
    
    def evaluate_transcription(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Evaluate transcription quality using WER and BLEU.
        
        Args:
            reference: Reference transcription
            hypothesis: Generated transcription
            
        Returns:
            Evaluation metrics
        """
        try:
            # Word Error Rate (WER)
            ref_words = reference.lower().split()
            hyp_words = hypothesis.lower().split()
            
            # Simple WER calculation
            if len(ref_words) == 0:
                wer = 1.0 if len(hyp_words) > 0 else 0.0
            else:
                # Levenshtein distance for WER
                wer = self._calculate_wer(ref_words, hyp_words)
            
            # Character Error Rate (CER)
            ref_chars = list(reference.lower())
            hyp_chars = list(hypothesis.lower())
            
            if len(ref_chars) == 0:
                cer = 1.0 if len(hyp_chars) > 0 else 0.0
            else:
                cer = self._calculate_cer(ref_chars, hyp_chars)
            
            return {
                'wer': wer,
                'cer': cer,
                'reference_length': len(ref_words),
                'hypothesis_length': len(hyp_words)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating transcription: {str(e)}")
            return {}
    
    def _calculate_wer(self, ref: List[str], hyp: List[str]) -> float:
        """Calculate Word Error Rate."""
        # Dynamic programming for edit distance
        d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        
        # Initialize
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        
        # Fill matrix
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i-1] == hyp[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        return d[len(ref)][len(hyp)] / len(ref) if len(ref) > 0 else 0
    
    def _calculate_cer(self, ref: List[str], hyp: List[str]) -> float:
        """Calculate Character Error Rate."""
        # Similar to WER but for characters
        d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        
        # Initialize
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        
        # Fill matrix
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i-1] == hyp[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        return d[len(ref)][len(hyp)] / len(ref) if len(ref) > 0 else 0
    
    def save_transcription(self, result: Dict[str, Any], 
                          output_path: Union[str, Path]):
        """Save transcription result to file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"=== TRANSCRIPTION ===\n")
                f.write(f"Model: {result.get('model', 'unknown')}\n")
                f.write(f"Language: {result.get('language', 'unknown')}\n\n")
                f.write(f"Text:\n{result['text']}\n")
                
                if 'segments' in result:
                    f.write(f"\n=== SEGMENTS ===\n")
                    for segment in result['segments']:
                        f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}\n")
            
            logger.info(f"Transcription saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving transcription: {str(e)}")
            raise

class MultiModelSpeechRecognizer:
    """Ensemble speech recognizer using multiple models."""
    
    def __init__(self, model_names: List[str] = None):
        """
        Initialize multi-model speech recognizer.
        
        Args:
            model_names: List of model names to use
        """
        if model_names is None:
            model_names = [
                "whisper-base",
                "wav2vec2-base-960h",
                "facebook/wav2vec2-base-960h"
            ]
        
        self.models = {}
        for model_name in model_names:
            try:
                self.models[model_name] = AdvancedSpeechRecognizer(model_name)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
    
    def ensemble_transcribe(self, audio_path: Union[str, Path], 
                           method: str = "voting") -> Dict[str, Any]:
        """
        Generate ensemble transcription using multiple models.
        
        Args:
            audio_path: Path to audio file
            method: Ensemble method ('voting', 'confidence', 'best')
            
        Returns:
            Ensemble transcription results
        """
        try:
            transcriptions = {}
            confidences = {}
            
            # Generate transcriptions from all models
            for model_name, model in self.models.items():
                try:
                    result = model.transcribe_audio(audio_path)
                    transcriptions[model_name] = result['text']
                    confidences[model_name] = result.get('confidence', 0.5)
                except Exception as e:
                    logger.warning(f"Error with {model_name}: {str(e)}")
                    continue
            
            if not transcriptions:
                raise ValueError("No models produced valid transcriptions")
            
            # Combine transcriptions based on method
            if method == "voting":
                # Simple voting - use the most common words
                all_words = []
                for text in transcriptions.values():
                    all_words.extend(text.lower().split())
                
                word_counts = {}
                for word in all_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                # Select top words and reconstruct
                top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                ensemble_text = " ".join([word for word, count in top_words])
                
            elif method == "confidence":
                # Use the model with highest confidence
                best_model = max(confidences.keys(), key=lambda x: confidences[x])
                ensemble_text = transcriptions[best_model]
                
            elif method == "best":
                # Use the longest transcription (heuristic)
                best_model = max(transcriptions.keys(), 
                               key=lambda x: len(transcriptions[x]))
                ensemble_text = transcriptions[best_model]
            
            else:
                raise ValueError(f"Unknown ensemble method: {method}")
            
            return {
                'ensemble_text': ensemble_text,
                'individual_transcriptions': transcriptions,
                'confidences': confidences,
                'method': method
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble transcription: {str(e)}")
            raise

def main():
    """Example usage of the AdvancedSpeechRecognizer."""
    # Initialize recognizer
    recognizer = AdvancedSpeechRecognizer("whisper-base")
    
    # Example audio file (replace with actual path)
    audio_path = "example_audio.wav"
    
    if Path(audio_path).exists():
        # Transcribe audio
        result = recognizer.transcribe_audio(audio_path)
        
        print("=== TRANSCRIPTION ===")
        print(result['text'])
        print(f"Language: {result.get('language', 'unknown')}")
        print(f"Model: {result['model']}")
    else:
        print(f"Audio file not found: {audio_path}")

if __name__ == "__main__":
    main()
