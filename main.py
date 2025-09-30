"""
Main application for the Video Summarizer project.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional
from utils.logger import setup_logger
from audioExtraction import AudioExtractor
from transcriptCreation import TranscriptGenerator
from transcriptSummariser import AdvancedSummarizer

logger = setup_logger(__name__)

class VideoSummarizer:
    """Main class for the Video Summarizer application."""
    
    def __init__(self):
        """Initialize the video summarizer."""
        self.audio_extractor = AudioExtractor()
        self.transcript_generator = TranscriptGenerator()
        self.summarizer = AdvancedSummarizer()
    
    def process_video(self, video_path: str, 
                     output_name: Optional[str] = None,
                     max_sentences: Optional[int] = None) -> dict:
        """
        Process a video file through the complete pipeline.
        
        Args:
            video_path: Path to the input video file
            output_name: Optional custom name for output files
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Dictionary with processing results and file paths
        """
        try:
            logger.info(f"Starting video processing: {video_path}")
            
            # Step 1: Extract audio
            logger.info("Step 1: Extracting audio...")
            audio_path = self.audio_extractor.extract_audio(video_path, output_name)
            logger.info(f"Audio extracted: {audio_path}")
            
            # Step 2: Generate transcript
            logger.info("Step 2: Generating transcript...")
            transcript_path = self.transcript_generator.transcribe_audio(audio_path, output_name)
            logger.info(f"Transcript generated: {transcript_path}")
            
            # Step 3: Generate summary
            logger.info("Step 3: Generating summary...")
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            
            summary_data = self.summarizer.generate_summary(transcript_text, max_sentences)
            summary_path = self.summarizer.save_summary(summary_data, output_name)
            logger.info(f"Summary generated: {summary_path}")
            
            # Return results
            results = {
                "video_path": video_path,
                "audio_path": str(audio_path),
                "transcript_path": str(transcript_path),
                "summary_path": str(summary_path),
                "summary_data": summary_data,
                "success": True
            }
            
            logger.info("Video processing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {
                "video_path": video_path,
                "error": str(e),
                "success": False
            }
    
    def process_audio(self, audio_path: str, 
                     output_name: Optional[str] = None,
                     max_sentences: Optional[int] = None) -> dict:
        """
        Process an audio file through the transcript and summary pipeline.
        
        Args:
            audio_path: Path to the input audio file
            output_name: Optional custom name for output files
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Dictionary with processing results and file paths
        """
        try:
            logger.info(f"Starting audio processing: {audio_path}")
            
            # Step 1: Generate transcript
            logger.info("Step 1: Generating transcript...")
            transcript_path = self.transcript_generator.transcribe_audio(audio_path, output_name)
            logger.info(f"Transcript generated: {transcript_path}")
            
            # Step 2: Generate summary
            logger.info("Step 2: Generating summary...")
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            
            summary_data = self.summarizer.generate_summary(transcript_text, max_sentences)
            summary_path = self.summarizer.save_summary(summary_data, output_name)
            logger.info(f"Summary generated: {summary_path}")
            
            # Return results
            results = {
                "audio_path": audio_path,
                "transcript_path": str(transcript_path),
                "summary_path": str(summary_path),
                "summary_data": summary_data,
                "success": True
            }
            
            logger.info("Audio processing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {
                "audio_path": audio_path,
                "error": str(e),
                "success": False
            }
    
    def process_transcript(self, transcript_path: str, 
                          output_name: Optional[str] = None,
                          max_sentences: Optional[int] = None) -> dict:
        """
        Process a transcript file to generate a summary.
        
        Args:
            transcript_path: Path to the input transcript file
            output_name: Optional custom name for output file
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Dictionary with processing results and file paths
        """
        try:
            logger.info(f"Starting transcript processing: {transcript_path}")
            
            # Read transcript
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            
            # Generate summary
            logger.info("Generating summary...")
            summary_data = self.summarizer.generate_summary(transcript_text, max_sentences)
            summary_path = self.summarizer.save_summary(summary_data, output_name)
            logger.info(f"Summary generated: {summary_path}")
            
            # Return results
            results = {
                "transcript_path": transcript_path,
                "summary_path": str(summary_path),
                "summary_data": summary_data,
                "success": True
            }
            
            logger.info("Transcript processing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error processing transcript: {str(e)}")
            return {
                "transcript_path": transcript_path,
                "error": str(e),
                "success": False
            }

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="AI-Powered Video Summarizer")
    parser.add_argument("input_file", help="Path to input video, audio, or transcript file")
    parser.add_argument("-o", "--output", help="Output name for generated files")
    parser.add_argument("-s", "--sentences", type=int, help="Maximum number of sentences in summary")
    parser.add_argument("-t", "--type", choices=["video", "audio", "transcript"], 
                       help="Input file type (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Initialize summarizer
    summarizer = VideoSummarizer()
    
    # Determine file type
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    file_type = args.type
    if file_type is None:
        # Auto-detect file type
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            file_type = "video"
        elif input_path.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']:
            file_type = "audio"
        elif input_path.suffix.lower() in ['.txt']:
            file_type = "transcript"
        else:
            print(f"Error: Unsupported file type: {input_path.suffix}")
            sys.exit(1)
    
    # Process file based on type
    print(f"Processing {file_type} file: {input_path}")
    
    if file_type == "video":
        results = summarizer.process_video(str(input_path), args.output, args.sentences)
    elif file_type == "audio":
        results = summarizer.process_audio(str(input_path), args.output, args.sentences)
    elif file_type == "transcript":
        results = summarizer.process_transcript(str(input_path), args.output, args.sentences)
    
    # Display results
    if results["success"]:
        print("\n=== PROCESSING COMPLETED ===")
        print(f"Summary saved to: {results['summary_path']}")
        
        if "summary_data" in results:
            print("\n=== SUMMARY ===")
            print(results["summary_data"]["summary"])
            
            if results["summary_data"]["action_items"]:
                print("\n=== ACTION ITEMS ===")
                for item in results["summary_data"]["action_items"]:
                    print(f"• {item}")
            
            if results["summary_data"]["keywords"]:
                print(f"\n=== KEYWORDS ===")
                print(", ".join(results["summary_data"]["keywords"][:10]))
    else:
        print(f"\nError: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
