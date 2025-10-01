"""
Video Source Manager for handling multiple video platforms.

This module provides comprehensive support for downloading videos from various platforms
including YouTube, Vimeo, Instagram, TikTok, Facebook, Twitter, Twitch, and many more.
"""

import re
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
import logging

# Try to import yt-dlp for video downloading
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

logger = logging.getLogger(__name__)

class VideoSourceManager:
    """Manages video downloads from multiple platforms using yt-dlp."""
    
    # Platform patterns for URL detection
    PLATFORM_PATTERNS = {
        'youtube': [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&\n?#]+)',
            r'(?:https?://)?(?:www\.)?youtu\.be/([^&\n?#]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([^&\n?#]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([^&\n?#]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([^&\n?#]+)'
        ],
        'vimeo': [
            r'(?:https?://)?(?:www\.)?vimeo\.com/(\d+)',
            r'(?:https?://)?(?:www\.)?vimeo\.com/channels/[^/]+/(\d+)',
            r'(?:https?://)?(?:www\.)?vimeo\.com/groups/[^/]+/videos/(\d+)',
            r'(?:https?://)?(?:www\.)?vimeo\.com/album/\d+/video/(\d+)'
        ],
        'instagram': [
            r'(?:https?://)?(?:www\.)?instagram\.com/p/([^/]+)',
            r'(?:https?://)?(?:www\.)?instagram\.com/reel/([^/]+)',
            r'(?:https?://)?(?:www\.)?instagram\.com/tv/([^/]+)'
        ],
        'tiktok': [
            r'(?:https?://)?(?:www\.)?tiktok\.com/@[^/]+/video/(\d+)',
            r'(?:https?://)?(?:vm\.)?tiktok\.com/([^/]+)'
        ],
        'facebook': [
            r'(?:https?://)?(?:www\.)?facebook\.com/[^/]+/videos/(\d+)',
            r'(?:https?://)?(?:www\.)?fb\.watch/([^/]+)'
        ],
        'twitter': [
            r'(?:https?://)?(?:www\.)?twitter\.com/[^/]+/status/(\d+)',
            r'(?:https?://)?(?:www\.)?x\.com/[^/]+/status/(\d+)',
            r'(?:https?://)?(?:www\.)?t\.co/([^/]+)'
        ],
        'twitch': [
            r'(?:https?://)?(?:www\.)?twitch\.tv/videos/(\d+)',
            r'(?:https?://)?(?:www\.)?twitch\.tv/[^/]+/clip/([^/]+)'
        ],
        'dailymotion': [
            r'(?:https?://)?(?:www\.)?dailymotion\.com/video/([^/]+)'
        ],
        'bilibili': [
            r'(?:https?://)?(?:www\.)?bilibili\.com/video/([^/]+)'
        ],
        'rumble': [
            r'(?:https?://)?(?:www\.)?rumble\.com/([^/]+)'
        ],
        'odysee': [
            r'(?:https?://)?(?:www\.)?odysee\.com/@[^/]+/[^/]+'
        ],
        'lbry': [
            r'(?:https?://)?(?:www\.)?lbry\.tv/@[^/]+/[^/]+'
        ]
    }
    
    # Platform-specific download configurations
    PLATFORM_CONFIGS = {
        'youtube': {
            'format': 'best[height<=720]',
            'noplaylist': True,
            'writesubtitles': False,
            'writeautomaticsub': False
        },
        'vimeo': {
            'format': 'best[height<=720]',
            'noplaylist': True
        },
        'instagram': {
            'format': 'best',
            'noplaylist': True
        },
        'tiktok': {
            'format': 'best',
            'noplaylist': True
        },
        'facebook': {
            'format': 'best[height<=720]',
            'noplaylist': True
        },
        'twitter': {
            'format': 'best[height<=720]',
            'noplaylist': True
        },
        'twitch': {
            'format': 'best[height<=720]',
            'noplaylist': True
        },
        'dailymotion': {
            'format': 'best[height<=720]',
            'noplaylist': True
        },
        'bilibili': {
            'format': 'best[height<=720]',
            'noplaylist': True
        },
        'rumble': {
            'format': 'best[height<=720]',
            'noplaylist': True
        },
        'odysee': {
            'format': 'best[height<=720]',
            'noplaylist': True
        },
        'lbry': {
            'format': 'best[height<=720]',
            'noplaylist': True
        }
    }
    
    def __init__(self, output_dir: str = "temp"):
        """
        Initialize the VideoSourceManager.
        
        Args:
            output_dir: Directory to save downloaded videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not YT_DLP_AVAILABLE:
            logger.warning("yt-dlp not available. Video downloading will not work.")
    
    def detect_platform(self, url: str) -> Optional[str]:
        """
        Detect the video platform from URL.
        
        Args:
            url: Video URL
            
        Returns:
            Platform name or None if not detected
        """
        url = url.strip()
        
        for platform, patterns in self.PLATFORM_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, url):
                    return platform
        
        return None
    
    def is_supported_url(self, url: str) -> bool:
        """
        Check if URL is supported by yt-dlp.
        
        Args:
            url: Video URL
            
        Returns:
            True if supported, False otherwise
        """
        if not YT_DLP_AVAILABLE:
            return False
        
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return info is not None
        except Exception:
            return False
    
    def get_platform_info(self, url: str) -> Dict[str, Any]:
        """
        Get information about the video without downloading.
        
        Args:
            url: Video URL
            
        Returns:
            Dictionary with video information
        """
        if not YT_DLP_AVAILABLE:
            return {'error': 'yt-dlp not available'}
        
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'platform': info.get('extractor_key', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'description': info.get('description', '')[:500] + '...' if info.get('description') else '',
                    'thumbnail': info.get('thumbnail', ''),
                    'url': url
                }
        except Exception as e:
            return {'error': str(e)}
    
    def download_video(self, url: str, custom_filename: Optional[str] = None) -> Optional[str]:
        """
        Download video from URL.
        
        Args:
            url: Video URL
            custom_filename: Optional custom filename
            
        Returns:
            Path to downloaded video or None if failed
        """
        if not YT_DLP_AVAILABLE:
            logger.error("yt-dlp not available for video downloading")
            return None
        
        try:
            # Detect platform
            platform = self.detect_platform(url)
            logger.info(f"Detected platform: {platform}")
            
            # Get platform-specific config
            config = self.PLATFORM_CONFIGS.get(platform, self.PLATFORM_CONFIGS['youtube'])
            
            # Use a simple filename template to avoid special character issues
            import time
            timestamp = int(time.time())
            outtmpl = str(self.output_dir / f"video_{timestamp}.%(ext)s")
            
            # Ensure the output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure yt-dlp options
            ydl_opts = {
                'outtmpl': outtmpl,
                'noplaylist': True,
                'no_warnings': False,
                'ignoreerrors': False,
                **config
            }
            
            # Add platform-specific options
            if platform == 'youtube':
                ydl_opts.update({
                    'writesubtitles': False,
                    'writeautomaticsub': False
                })
            elif platform in ['instagram', 'tiktok']:
                # These platforms often have shorter videos, use best quality
                ydl_opts['format'] = 'best'
            
            logger.info(f"Downloading video from {platform}: {url}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'Unknown')
                logger.info(f"Video title: {video_title}")
                
                # Download the video
                ydl.download([url])
                
                # Find the downloaded file
                for file in self.output_dir.iterdir():
                    if file.suffix.lower() in ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv']:
                        logger.info(f"Video downloaded successfully: {file}")
                        return str(file)
                
                logger.error("Downloaded file not found")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            # Try with an even simpler filename as fallback
            try:
                logger.info("Trying fallback download with simpler filename...")
                fallback_outtmpl = str(self.output_dir / "video.%(ext)s")
                ydl_opts['outtmpl'] = fallback_outtmpl
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                    
                    # Find the downloaded file
                    for file in self.output_dir.iterdir():
                        if file.suffix.lower() in ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv']:
                            logger.info(f"Video downloaded successfully with fallback: {file}")
                            return str(file)
                    
                    logger.error("Fallback download also failed")
                    return None
                    
            except Exception as fallback_error:
                logger.error(f"Fallback download also failed: {str(fallback_error)}")
                return None
    
    def get_supported_platforms(self) -> List[str]:
        """
        Get list of supported platforms.
        
        Returns:
            List of supported platform names
        """
        return list(self.PLATFORM_PATTERNS.keys())
    
    def validate_url(self, url: str) -> Tuple[bool, str]:
        """
        Validate URL and return status with message.
        
        Args:
            url: Video URL to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not url.strip():
            return False, "URL cannot be empty"
        
        # Check if it's a valid URL format
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False, "Invalid URL format"
        except Exception:
            return False, "Invalid URL format"
        
        # Check if platform is detected
        platform = self.detect_platform(url)
        if not platform:
            return False, f"Unsupported platform. Supported platforms: {', '.join(self.get_supported_platforms())}"
        
        # Check if yt-dlp can handle it
        if not self.is_supported_url(url):
            return False, f"URL not supported by yt-dlp or video is not accessible"
        
        return True, f"Valid {platform} URL"
    
    def get_platform_icon(self, platform: str) -> str:
        """
        Get emoji icon for platform.
        
        Args:
            platform: Platform name
            
        Returns:
            Emoji icon string
        """
        icons = {
            'youtube': '📺',
            'vimeo': '🎬',
            'instagram': '📷',
            'tiktok': '🎵',
            'facebook': '👥',
            'twitter': '🐦',
            'twitch': '🎮',
            'dailymotion': '🎥',
            'bilibili': '🇨🇳',
            'rumble': '⚡',
            'odysee': '🔍',
            'lbry': '🔗'
        }
        return icons.get(platform, '🎥')
    
    def get_platform_description(self, platform: str) -> str:
        """
        Get description for platform.
        
        Args:
            platform: Platform name
            
        Returns:
            Platform description
        """
        descriptions = {
            'youtube': 'YouTube - The world\'s largest video platform',
            'vimeo': 'Vimeo - High-quality video hosting and sharing',
            'instagram': 'Instagram - Photo and video sharing platform',
            'tiktok': 'TikTok - Short-form video platform',
            'facebook': 'Facebook - Social media video content',
            'twitter': 'Twitter/X - Social media video posts',
            'twitch': 'Twitch - Live streaming and gaming content',
            'dailymotion': 'Dailymotion - Video sharing platform',
            'bilibili': 'Bilibili - Chinese video platform',
            'rumble': 'Rumble - Video platform and creator network',
            'odysee': 'Odysee - Decentralized video platform',
            'lbry': 'LBRY - Decentralized content platform'
        }
        return descriptions.get(platform, 'Unknown platform')
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """
        Clean up old downloaded files.
        
        Args:
            max_age_hours: Maximum age of files in hours
        """
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file in self.output_dir.iterdir():
            if file.is_file() and current_time - file.stat().st_mtime > max_age_seconds:
                try:
                    file.unlink()
                    logger.info(f"Cleaned up old file: {file}")
                except Exception as e:
                    logger.warning(f"Could not delete file {file}: {e}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing/replacing invalid characters."""
        import re
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Replace multiple spaces with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')
        # Limit length to avoid filesystem issues
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized


def create_video_source_manager(output_dir: str = "temp") -> VideoSourceManager:
    """
    Factory function to create VideoSourceManager instance.
    
    Args:
        output_dir: Directory for downloaded videos
        
    Returns:
        VideoSourceManager instance
    """
    return VideoSourceManager(output_dir)
