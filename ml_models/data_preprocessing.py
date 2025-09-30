"""
Advanced data preprocessing and feature engineering pipelines for video summarization.
"""
import torch
import torchaudio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import cv2
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from collections import Counter
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from config import OUTPUT_DIR

logger = setup_logger(__name__)

class AudioPreprocessor:
    """Advanced audio preprocessing pipeline."""
    
    def __init__(self, target_sr: int = 16000, max_duration: float = 300.0):
        """
        Initialize audio preprocessor.
        
        Args:
            target_sr: Target sample rate
            max_duration: Maximum duration in seconds
        """
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.scaler = StandardScaler()
    
    def extract_audio(self, video_path: Union[str, Path], 
                     output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Extract audio from video file using multiple methods.
        
        Args:
            video_path: Path to video file
            output_path: Optional output path for audio
            
        Returns:
            Path to extracted audio file
        """
        try:
            video_path = Path(video_path)
            if output_path is None:
                # Create a safe filename without special characters
                safe_name = "".join(c for c in video_path.stem if c.isalnum() or c in (' ', '-', '_')).rstrip()
                output_path = video_path.parent / f"{safe_name}_audio.wav"
            else:
                output_path = Path(output_path)
            
            logger.info(f"Extracting audio from video: {video_path}")
            
            # Try multiple audio extraction methods
            # Method 1: Try librosa with different backends
            try:
                audio, sr = librosa.load(str(video_path), sr=16000, mono=True)
                logger.info(f"Audio loaded with librosa: shape={audio.shape}, sr={sr}")
                sf.write(str(output_path), audio, sr)
                logger.info(f"Audio saved to: {output_path}")
                return str(output_path)
            except Exception as e:
                logger.warning(f"Librosa extraction failed: {str(e)}")
            
            # Method 2: Try using moviepy (if available)
            try:
                from moviepy.editor import VideoFileClip
                video = VideoFileClip(str(video_path))
                video.audio.write_audiofile(str(output_path), verbose=False, logger=None)
                video.close()
                logger.info(f"Audio extracted successfully with moviepy: {output_path}")
                return str(output_path)
            except ImportError:
                logger.warning("MoviePy not available for audio extraction")
            except Exception as e:
                logger.warning(f"MoviePy extraction failed: {str(e)}")
            
            # Method 3: Try using pydub (if available)
            try:
                from pydub import AudioSegment
                from pydub.utils import which
                
                # Check if ffmpeg is available
                if which("ffmpeg"):
                    audio = AudioSegment.from_file(str(video_path))
                    audio.export(str(output_path), format="wav")
                    logger.info(f"Audio extracted successfully with pydub: {output_path}")
                    return str(output_path)
                else:
                    logger.warning("FFmpeg not available for pydub audio extraction")
            except ImportError:
                logger.warning("Pydub not available for audio extraction")
            except Exception as e:
                logger.warning(f"Pydub extraction failed: {str(e)}")
            
            # Method 4: Create a dummy audio file for unsupported formats
            logger.warning("All audio extraction methods failed, creating dummy audio")
            
            # Create a short silence audio file
            duration = 5.0  # 5 seconds
            sr = 16000
            silence = np.zeros(int(duration * sr), dtype=np.float32)
            sf.write(str(output_path), silence, sr)
            
            logger.info(f"Created dummy audio file: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error in audio extraction: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _extract_audio_librosa(self, video_path: Union[str, Path], 
                              output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Extract audio using librosa as fallback.
        
        Args:
            video_path: Path to video file
            output_path: Optional output path for audio
            
        Returns:
            Path to extracted audio file
        """
        try:
            video_path = Path(video_path)
            if output_path is None:
                safe_name = "".join(c for c in video_path.stem if c.isalnum() or c in (' ', '-', '_')).rstrip()
                output_path = video_path.parent / f"{safe_name}_audio.wav"
            else:
                output_path = Path(output_path)
            
            # Load audio with librosa
            audio, sr = librosa.load(str(video_path), sr=16000, mono=True)
            
            # Save as WAV
            sf.write(str(output_path), audio, sr)
            
            logger.info(f"Audio extracted with librosa: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Librosa audio extraction failed: {str(e)}")
            raise
        
    def preprocess_audio(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio data
        """
        try:
            logger.info(f"Loading audio from: {audio_path}")
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.target_sr)
            logger.info(f"Audio loaded: shape={audio.shape}, sr={sr}")
            
            # Check if audio is empty
            if len(audio) == 0:
                logger.warning("Audio is empty, creating silent audio")
                audio = np.zeros(int(self.target_sr * 1.0))  # 1 second of silence
                sr = self.target_sr
            
            # Trim silence (but don't fail if it doesn't work)
            try:
                audio, _ = librosa.effects.trim(audio, top_db=20)
                logger.info(f"Audio trimmed: shape={audio.shape}")
            except Exception as trim_error:
                logger.warning(f"Could not trim audio: {trim_error}")
            
            # Normalize
            try:
                audio = librosa.util.normalize(audio)
                logger.info(f"Audio normalized: shape={audio.shape}")
            except Exception as norm_error:
                logger.warning(f"Could not normalize audio: {norm_error}")
            
            # Pad or truncate to max_duration
            max_samples = int(self.max_duration * self.target_sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
            logger.info(f"Audio padded/truncated: shape={audio.shape}")
            
            # Extract features (simplified)
            logger.info("Extracting audio features...")
            try:
                features = self._extract_audio_features(audio, sr)
                logger.info(f"Audio features extracted: {list(features.keys())}")
            except Exception as feat_error:
                logger.warning(f"Could not extract all features: {feat_error}")
                # Create basic features
                features = {
                    'mfcc': np.zeros((13, 1)),
                    'spectral_centroid': np.zeros((1,)),
                    'rms': np.zeros((1,)),
                    'statistics': {'mean': 0.0, 'std': 0.0, 'energy': 0.0}
                }
            
            return {
                'audio': audio,
                'sample_rate': sr,
                'duration': len(audio) / sr,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return basic structure even if processing fails
            return {
                'audio': np.zeros(int(self.target_sr * 1.0)),
                'sample_rate': self.target_sr,
                'duration': 1.0,
                'features': {'mfcc': np.zeros((13, 1)), 'statistics': {'mean': 0.0, 'std': 0.0, 'energy': 0.0}}
            }
    
    def _extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features with error handling."""
        features = {}
        
        try:
            # Basic spectral features (most reliable)
            features['mfcc'] = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
            features['rms'] = librosa.feature.rms(y=audio)
        except Exception as e:
            logger.warning(f"Error with basic features: {e}")
            # Fallback to simple features
            features['mfcc'] = np.zeros((13, 1))
            features['spectral_centroid'] = np.zeros((1,))
            features['zero_crossing_rate'] = np.zeros((1,))
            features['rms'] = np.zeros((1,))
        
        try:
            # Additional features (may fail)
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        except:
            features['spectral_rolloff'] = np.zeros((1,))
        
        try:
            features['tempo'] = librosa.beat.tempo(y=audio, sr=sr)
        except:
            features['tempo'] = np.array([0.0])
        
        try:
            features['chroma'] = librosa.feature.chroma_stft(y=audio, sr=sr)
        except:
            features['chroma'] = np.zeros((12, 1))
        
        try:
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        except:
            features['spectral_bandwidth'] = np.zeros((1,))
        
        # Statistical features (always try)
        try:
            features['statistics'] = self._calculate_audio_statistics(audio)
        except Exception as e:
            logger.warning(f"Error calculating statistics: {e}")
            features['statistics'] = {'mean': 0.0, 'std': 0.0, 'energy': 0.0}
        
        return features
    
    def _calculate_audio_statistics(self, audio: np.ndarray) -> Dict[str, float]:
        """Calculate statistical features of audio."""
        return {
            'mean': np.mean(audio),
            'std': np.std(audio),
            'min': np.min(audio),
            'max': np.max(audio),
            'skewness': self._calculate_skewness(audio),
            'kurtosis': self._calculate_kurtosis(audio),
            'energy': np.sum(audio ** 2),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio))
        }
    
    def _calculate_skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.mean(((x - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis."""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.mean(((x - mean) / std) ** 4) - 3

class VideoPreprocessor:
    """Advanced video preprocessing pipeline."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), 
                 max_frames: int = 100):
        """
        Initialize video preprocessor.
        
        Args:
            target_size: Target frame size (height, width)
            max_frames: Maximum number of frames to extract
        """
        self.target_size = target_size
        self.max_frames = max_frames
        
    def preprocess_video(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Preprocess video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed video data
        """
        try:
            # Extract frames
            frames = self._extract_frames(video_path)
            
            # Resize frames
            frames = [self._resize_frame(frame) for frame in frames]
            
            # Extract features
            features = self._extract_video_features(frames)
            
            return {
                'frames': frames,
                'frame_count': len(frames),
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing video: {str(e)}")
            raise
    
    def _extract_frames(self, video_path: Union[str, Path]) -> List[np.ndarray]:
        """Extract frames from video."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // self.max_frames)
            
            frame_count = 0
            while cap.isOpened() and len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target size."""
        return cv2.resize(frame, (self.target_size[1], self.target_size[0]))
    
    def _extract_video_features(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Extract video features."""
        try:
            features = {}
            
            if not frames:
                return features
            
            # Convert frames to numpy array
            frames_array = np.array(frames)
            
            # Color features
            features['color_histogram'] = self._calculate_color_histogram(frames_array)
            features['color_mean'] = np.mean(frames_array, axis=(0, 1, 2))
            features['color_std'] = np.std(frames_array, axis=(0, 1, 2))
            
            # Motion features
            features['motion_vectors'] = self._calculate_motion_vectors(frames_array)
            
            # Texture features
            features['texture_features'] = self._calculate_texture_features(frames_array)
            
            # Statistical features
            features['statistics'] = self._calculate_video_statistics(frames_array)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting video features: {str(e)}")
            return {}
    
    def _calculate_color_histogram(self, frames: np.ndarray) -> np.ndarray:
        """Calculate color histogram for frames."""
        histograms = []
        for frame in frames:
            hist_r = cv2.calcHist([frame[:,:,0]], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([frame[:,:,1]], [0], None, [256], [0, 256])
            hist_b = cv2.calcHist([frame[:,:,2]], [0], None, [256], [0, 256])
            hist = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
            histograms.append(hist)
        
        return np.mean(histograms, axis=0)
    
    def _calculate_motion_vectors(self, frames: np.ndarray) -> np.ndarray:
        """Calculate motion vectors between consecutive frames."""
        if len(frames) < 2:
            return np.array([])
        
        motion_vectors = []
        for i in range(1, len(frames)):
            # Convert to grayscale
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                gray1, gray2,
                np.array([[100, 100]], dtype=np.float32),
                None
            )[0]
            
            if flow is not None and len(flow) > 0:
                motion_vectors.append(flow[0])
            else:
                motion_vectors.append([0, 0])
        
        return np.array(motion_vectors)
    
    def _calculate_texture_features(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate texture features using LBP."""
        try:
            from skimage.feature import local_binary_pattern
            
            lbp_features = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                lbp = local_binary_pattern(gray, 8, 1, method='uniform')
                hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
                lbp_features.append(hist)
            
            return {
                'lbp_histogram': np.mean(lbp_features, axis=0),
                'lbp_mean': np.mean(lbp_features),
                'lbp_std': np.std(lbp_features)
            }
        except ImportError:
            logger.warning("scikit-image not available for texture features")
            return {}
        except Exception as e:
            logger.warning(f"Error calculating texture features: {e}")
            return {}
    
    def _calculate_video_statistics(self, frames: np.ndarray) -> Dict[str, float]:
        """Calculate statistical features of video."""
        return {
            'brightness_mean': np.mean(frames),
            'brightness_std': np.std(frames),
            'contrast_mean': np.mean(np.std(frames, axis=(1, 2))),
            'contrast_std': np.std(np.std(frames, axis=(1, 2)))
        }

class TextPreprocessor:
    """Advanced text preprocessing pipeline."""
    
    def __init__(self, language: str = "english"):
        """
        Initialize text preprocessor.
        
        Args:
            language: Language for text processing
        """
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
        # Download required NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Preprocess text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text data
        """
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Tokenize
            sentences = sent_tokenize(cleaned_text)
            words = word_tokenize(cleaned_text)
            
            # Remove stopwords
            filtered_words = [word for word in words if word.lower() not in self.stop_words]
            
            # Stem and lemmatize
            stemmed_words = [self.stemmer.stem(word) for word in filtered_words]
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in filtered_words]
            
            # Extract features
            features = self._extract_text_features(cleaned_text, sentences, words)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'sentences': sentences,
                'words': words,
                'filtered_words': filtered_words,
                'stemmed_words': stemmed_words,
                'lemmatized_words': lemmatized_words,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and normalizing."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        return text.strip()
    
    def _extract_text_features(self, text: str, sentences: List[str], 
                              words: List[str]) -> Dict[str, Any]:
        """Extract comprehensive text features."""
        try:
            features = {}
            
            # Basic statistics
            features['char_count'] = len(text)
            features['word_count'] = len(words)
            features['sentence_count'] = len(sentences)
            features['avg_word_length'] = np.mean([len(word) for word in words])
            features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences])
            
            # Readability features
            features['readability'] = self._calculate_readability(text)
            
            # Sentiment features
            features['sentiment'] = self._calculate_sentiment(text)
            
            # POS tagging features
            features['pos_tags'] = self._extract_pos_features(words)
            
            # N-gram features
            features['bigrams'] = self._extract_ngrams(words, 2)
            features['trigrams'] = self._extract_ngrams(words, 3)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting text features: {str(e)}")
            return {}
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics."""
        try:
            import textstat
            
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'smog_index': textstat.smog_index(text)
            }
        except ImportError:
            logger.warning("textstat not available for readability metrics")
            return {}
        except Exception as e:
            logger.warning(f"Error calculating readability: {e}")
            return {}
    
    def _calculate_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores."""
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except ImportError:
            logger.warning("textblob not available for sentiment analysis")
            return {}
        except Exception as e:
            logger.warning(f"Error calculating sentiment: {e}")
            return {}
    
    def _extract_pos_features(self, words: List[str]) -> Dict[str, int]:
        """Extract POS tag features."""
        try:
            pos_tags = nltk.pos_tag(words)
            tag_counts = Counter([tag for word, tag in pos_tags])
            
            return dict(tag_counts)
        except Exception as e:
            logger.warning(f"Error extracting POS features: {str(e)}")
            return {}
    
    def _extract_ngrams(self, words: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from words."""
        if len(words) < n:
            return []
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i+n]))
        
        return ngrams

class FeatureEngineer:
    """Advanced feature engineering for ML models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.vectorizers = {}
        self.scalers = {}
        self.encoders = {}
        self.dimensionality_reducers = {}
        
    def create_text_features(self, texts: List[str], 
                           feature_type: str = "tfidf",
                           max_features: int = 1000) -> np.ndarray:
        """
        Create text features using various vectorization methods.
        
        Args:
            texts: List of texts
            feature_type: Type of features ('tfidf', 'count', 'embedding')
            max_features: Maximum number of features
            
        Returns:
            Feature matrix
        """
        try:
            if feature_type == "tfidf":
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            elif feature_type == "count":
                vectorizer = CountVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            elif feature_type == "embedding":
                # Use sentence transformers for embeddings
                model = SentenceTransformer('all-MiniLM-L6-v2')
                return model.encode(texts)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            features = vectorizer.fit_transform(texts).toarray()
            self.vectorizers[feature_type] = vectorizer
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating text features: {str(e)}")
            raise
    
    def create_audio_features(self, audio_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create audio features from preprocessed audio data.
        
        Args:
            audio_data: List of preprocessed audio data
            
        Returns:
            Feature matrix
        """
        try:
            features = []
            
            for audio in audio_data:
                audio_features = []
                
                # Extract MFCC features
                if 'mfcc' in audio['features']:
                    mfcc = audio['features']['mfcc']
                    audio_features.extend([
                        np.mean(mfcc, axis=1),
                        np.std(mfcc, axis=1),
                        np.max(mfcc, axis=1),
                        np.min(mfcc, axis=1)
                    ])
                
                # Extract spectral features
                for feature_name in ['spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate']:
                    if feature_name in audio['features']:
                        feature = audio['features'][feature_name]
                        audio_features.extend([
                            np.mean(feature),
                            np.std(feature),
                            np.max(feature),
                            np.min(feature)
                        ])
                
                # Extract statistical features
                if 'statistics' in audio['features']:
                    stats = audio['features']['statistics']
                    audio_features.extend([
                        stats['mean'],
                        stats['std'],
                        stats['energy'],
                        stats['zero_crossing_rate']
                    ])
                
                features.append(np.concatenate(audio_features))
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error creating audio features: {str(e)}")
            raise
    
    def create_video_features(self, video_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create video features from preprocessed video data.
        
        Args:
            video_data: List of preprocessed video data
            
        Returns:
            Feature matrix
        """
        try:
            features = []
            
            for video in video_data:
                video_features = []
                
                # Extract color features
                if 'color_histogram' in video['features']:
                    video_features.extend(video['features']['color_histogram'])
                
                if 'color_mean' in video['features']:
                    video_features.extend(video['features']['color_mean'])
                
                if 'color_std' in video['features']:
                    video_features.extend(video['features']['color_std'])
                
                # Extract motion features
                if 'motion_vectors' in video['features']:
                    motion = video['features']['motion_vectors']
                    if len(motion) > 0:
                        video_features.extend([
                            np.mean(motion, axis=0),
                            np.std(motion, axis=0),
                            np.max(motion, axis=0),
                            np.min(motion, axis=0)
                        ])
                
                # Extract statistical features
                if 'statistics' in video['features']:
                    stats = video['features']['statistics']
                    video_features.extend([
                        stats['brightness_mean'],
                        stats['brightness_std'],
                        stats['contrast_mean'],
                        stats['contrast_std']
                    ])
                
                features.append(np.concatenate(video_features))
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error creating video features: {str(e)}")
            raise
    
    def scale_features(self, features: np.ndarray, 
                      scaler_type: str = "standard",
                      feature_name: str = "default") -> np.ndarray:
        """
        Scale features using various scaling methods.
        
        Args:
            features: Feature matrix
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            feature_name: Name for the scaler
            
        Returns:
            Scaled features
        """
        try:
            if scaler_type == "standard":
                scaler = StandardScaler()
            elif scaler_type == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
            
            scaled_features = scaler.fit_transform(features)
            self.scalers[feature_name] = scaler
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise
    
    def reduce_dimensionality(self, features: np.ndarray,
                            method: str = "pca",
                            n_components: int = 50,
                            feature_name: str = "default") -> np.ndarray:
        """
        Reduce dimensionality of features.
        
        Args:
            features: Feature matrix
            method: Dimensionality reduction method ('pca', 'svd', 'tsne')
            n_components: Number of components
            feature_name: Name for the reducer
            
        Returns:
            Reduced features
        """
        try:
            if method == "pca":
                reducer = PCA(n_components=n_components)
            elif method == "svd":
                reducer = TruncatedSVD(n_components=n_components)
            else:
                raise ValueError(f"Unknown reduction method: {method}")
            
            reduced_features = reducer.fit_transform(features)
            self.dimensionality_reducers[feature_name] = reducer
            
            return reduced_features
            
        except Exception as e:
            logger.error(f"Error reducing dimensionality: {str(e)}")
            raise
    
    def create_cluster_features(self, features: np.ndarray,
                              n_clusters: int = 5,
                              feature_name: str = "default") -> np.ndarray:
        """
        Create cluster-based features.
        
        Args:
            features: Feature matrix
            n_clusters: Number of clusters
            feature_name: Name for the clusterer
            
        Returns:
            Cluster features
        """
        try:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(features)
            
            # Create one-hot encoded cluster features
            cluster_features = np.zeros((len(features), n_clusters))
            cluster_features[np.arange(len(features)), cluster_labels] = 1
            
            self.encoders[feature_name] = clusterer
            
            return cluster_features
            
        except Exception as e:
            logger.error(f"Error creating cluster features: {str(e)}")
            raise
    
    def save_preprocessors(self, save_path: Union[str, Path]):
        """Save all preprocessors and feature engineers."""
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save vectorizers
            for name, vectorizer in self.vectorizers.items():
                joblib.dump(vectorizer, save_path / f"vectorizer_{name}.pkl")
            
            # Save scalers
            for name, scaler in self.scalers.items():
                joblib.dump(scaler, save_path / f"scaler_{name}.pkl")
            
            # Save encoders
            for name, encoder in self.encoders.items():
                joblib.dump(encoder, save_path / f"encoder_{name}.pkl")
            
            # Save dimensionality reducers
            for name, reducer in self.dimensionality_reducers.items():
                joblib.dump(reducer, save_path / f"reducer_{name}.pkl")
            
            logger.info(f"Preprocessors saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessors: {str(e)}")
            raise
    
    def load_preprocessors(self, load_path: Union[str, Path]):
        """Load preprocessors and feature engineers."""
        try:
            load_path = Path(load_path)
            
            # Load vectorizers
            for file_path in load_path.glob("vectorizer_*.pkl"):
                name = file_path.stem.replace("vectorizer_", "")
                self.vectorizers[name] = joblib.load(file_path)
            
            # Load scalers
            for file_path in load_path.glob("scaler_*.pkl"):
                name = file_path.stem.replace("scaler_", "")
                self.scalers[name] = joblib.load(file_path)
            
            # Load encoders
            for file_path in load_path.glob("encoder_*.pkl"):
                name = file_path.stem.replace("encoder_", "")
                self.encoders[name] = joblib.load(file_path)
            
            # Load dimensionality reducers
            for file_path in load_path.glob("reducer_*.pkl"):
                name = file_path.stem.replace("reducer_", "")
                self.dimensionality_reducers[name] = joblib.load(file_path)
            
            logger.info(f"Preprocessors loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise

def main():
    """Example usage of the preprocessing pipeline."""
    # Initialize preprocessors
    audio_preprocessor = AudioPreprocessor()
    video_preprocessor = VideoPreprocessor()
    text_preprocessor = TextPreprocessor()
    feature_engineer = FeatureEngineer()
    
    # Example text preprocessing
    text = "This is an example text for preprocessing. It contains multiple sentences and various features."
    text_data = text_preprocessor.preprocess_text(text)
    
    print("Text preprocessing completed:")
    print(f"Original words: {len(text_data['words'])}")
    print(f"Filtered words: {len(text_data['filtered_words'])}")
    print(f"Features: {list(text_data['features'].keys())}")

if __name__ == "__main__":
    main()
