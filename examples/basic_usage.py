"""
Basic usage examples for the Video Summarizer.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main import VideoSummarizer
from email_sender import EmailSender

def example_video_processing():
    """Example of processing a video file."""
    print("=== Video Processing Example ===")
    
    # Initialize summarizer
    summarizer = VideoSummarizer()
    
    # Process video (replace with your video path)
    video_path = "sample_meeting.mp4"
    
    try:
        results = summarizer.process_video(
            video_path=video_path,
            output_name="team_meeting",
            max_sentences=5
        )
        
        if results["success"]:
            print("✅ Video processed successfully!")
            print(f"📄 Transcript: {results['transcript_path']}")
            print(f"📋 Summary: {results['summary_path']}")
            
            # Display summary
            summary_data = results["summary_data"]
            print(f"\n📝 Summary:\n{summary_data['summary']}")
            
            if summary_data["action_items"]:
                print(f"\n🎯 Action Items:")
                for i, item in enumerate(summary_data["action_items"], 1):
                    print(f"  {i}. {item}")
            
            if summary_data["keywords"]:
                print(f"\n🔑 Keywords: {', '.join(summary_data['keywords'][:10])}")
                
        else:
            print(f"❌ Error: {results['error']}")
            
    except FileNotFoundError:
        print(f"❌ Video file not found: {video_path}")
        print("Please provide a valid video file path")

def example_audio_processing():
    """Example of processing an audio file."""
    print("\n=== Audio Processing Example ===")
    
    summarizer = VideoSummarizer()
    audio_path = "sample_audio.wav"
    
    try:
        results = summarizer.process_audio(
            audio_path=audio_path,
            output_name="audio_meeting",
            max_sentences=3
        )
        
        if results["success"]:
            print("✅ Audio processed successfully!")
            print(f"📄 Transcript: {results['transcript_path']}")
            print(f"📋 Summary: {results['summary_path']}")
        else:
            print(f"❌ Error: {results['error']}")
            
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_path}")

def example_transcript_processing():
    """Example of processing a transcript file."""
    print("\n=== Transcript Processing Example ===")
    
    summarizer = VideoSummarizer()
    transcript_path = "sample_transcript.txt"
    
    try:
        results = summarizer.process_transcript(
            transcript_path=transcript_path,
            output_name="transcript_summary",
            max_sentences=4
        )
        
        if results["success"]:
            print("✅ Transcript processed successfully!")
            print(f"📋 Summary: {results['summary_path']}")
        else:
            print(f"❌ Error: {results['error']}")
            
    except FileNotFoundError:
        print(f"❌ Transcript file not found: {transcript_path}")

def example_email_sending():
    """Example of sending summary via email."""
    print("\n=== Email Sending Example ===")
    
    # Configure email (you'll need to set up your email credentials)
    sender = EmailSender(
        sender_email="your_email@gmail.com",
        sender_password="your_app_password"
    )
    
    # Sample summary data
    summary_data = {
        "summary": "This is a sample meeting summary with key discussion points.",
        "action_items": [
            "Follow up on project proposal by Friday",
            "Schedule next team meeting for next week",
            "Review budget allocation for Q2"
        ],
        "keywords": ["project", "meeting", "proposal", "budget", "team"],
        "named_entities": [
            ("John Smith", "PERSON"),
            ("Acme Corporation", "ORG"),
            ("Q2 2024", "DATE")
        ],
        "metadata": {
            "original_sentence_count": 25,
            "summary_sentence_count": 3,
            "compression_ratio": 0.12,
            "action_item_count": 3,
            "keyword_count": 5,
            "named_entity_count": 3
        }
    }
    
    # Send email
    recipients = ["participant1@example.com", "participant2@example.com"]
    
    print("Note: This example requires email configuration.")
    print("To test email functionality:")
    print("1. Set up your email credentials in config.py")
    print("2. Uncomment the lines below")
    
    # Uncomment to test email sending
    # success = sender.send_summary_email(
    #     recipients=recipients,
    #     summary_data=summary_data,
    #     meeting_title="Weekly Team Meeting"
    # )
    # 
    # if success:
    #     print("✅ Email sent successfully!")
    # else:
    #     print("❌ Failed to send email")

def main():
    """Run all examples."""
    print("🎥 Video Summarizer - Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_video_processing()
    example_audio_processing()
    example_transcript_processing()
    example_email_sending()
    
    print("\n" + "=" * 50)
    print("✅ Examples completed!")
    print("\nTo run the web interface:")
    print("  streamlit run streamlit_app.py")
    print("\nTo run the CLI:")
    print("  python main.py your_video.mp4")

if __name__ == "__main__":
    main()
