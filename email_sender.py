"""
Email functionality for sending summaries to meeting participants.
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import List, Dict, Any, Optional
from utils.logger import setup_logger
from config import EMAIL_SETTINGS

logger = setup_logger(__name__)

class EmailSender:
    """Handles sending email summaries to meeting participants."""
    
    def __init__(self, smtp_server: Optional[str] = None, 
                 smtp_port: Optional[int] = None,
                 sender_email: Optional[str] = None,
                 sender_password: Optional[str] = None):
        """
        Initialize the email sender.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address
            sender_password: Sender email password
        """
        self.smtp_server = smtp_server or EMAIL_SETTINGS["smtp_server"]
        self.smtp_port = smtp_port or EMAIL_SETTINGS["smtp_port"]
        self.sender_email = sender_email or EMAIL_SETTINGS["sender_email"]
        self.sender_password = sender_password or EMAIL_SETTINGS["sender_password"]
        self.use_tls = EMAIL_SETTINGS["use_tls"]
    
    def create_summary_email(self, summary_data: Dict[str, Any], 
                           meeting_title: str = "Meeting Summary",
                           participants: List[str] = None) -> str:
        """
        Create HTML email content for the summary.
        
        Args:
            summary_data: Summary data dictionary
            meeting_title: Title of the meeting
            participants: List of participant email addresses
            
        Returns:
            HTML email content
        """
        action_items = summary_data.get('action_items', [])
        keywords = summary_data.get('keywords', [])
        named_entities = summary_data.get('named_entities', [])
        metadata = summary_data.get('metadata', {})
        
        # Create action items HTML
        action_items_html = ""
        if action_items:
            action_items_html = """
            <h3>🎯 Action Items</h3>
            <ul>
            """
            for i, item in enumerate(action_items, 1):
                action_items_html += f"<li>{item}</li>"
            action_items_html += "</ul>"
        
        # Create keywords HTML
        keywords_html = ""
        if keywords:
            keywords_html = f"""
            <h3>🔑 Key Topics</h3>
            <p><strong>{', '.join(keywords[:10])}</strong></p>
            """
        
        # Create named entities HTML
        entities_html = ""
        if named_entities:
            entities_html = """
            <h3>👥 People & Organizations Mentioned</h3>
            <ul>
            """
            for entity, label in named_entities[:10]:  # Limit to 10 entities
                entities_html += f"<li><strong>{entity}</strong> ({label})</li>"
            entities_html += "</ul>"
        
        # Create participants HTML
        participants_html = ""
        if participants:
            participants_html = f"""
            <h3>📧 Sent to:</h3>
            <p>{', '.join(participants)}</p>
            """
        
        # Create metadata HTML
        metadata_html = f"""
        <h3>📊 Summary Statistics</h3>
        <ul>
            <li>Original content: {metadata.get('original_sentence_count', 0)} sentences</li>
            <li>Summary: {metadata.get('summary_sentence_count', 0)} sentences</li>
            <li>Compression ratio: {metadata.get('compression_ratio', 0):.1%}</li>
            <li>Action items found: {metadata.get('action_item_count', 0)}</li>
            <li>Keywords identified: {metadata.get('keyword_count', 0)}</li>
        </ul>
        """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .summary-box {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 4px solid #667eea;
                    margin: 20px 0;
                }}
                .action-item {{
                    background: #e3f2fd;
                    padding: 10px;
                    border-radius: 8px;
                    margin: 10px 0;
                    border-left: 3px solid #2196f3;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                ul {{
                    padding-left: 20px;
                }}
                .footer {{
                    text-align: center;
                    color: #666;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📋 {meeting_title}</h1>
                <p>AI-Generated Meeting Summary</p>
            </div>
            
            <div class="summary-box">
                <h2>📝 Summary</h2>
                <p>{summary_data.get('summary', 'No summary available')}</p>
            </div>
            
            {action_items_html}
            {keywords_html}
            {entities_html}
            {participants_html}
            {metadata_html}
            
            <div class="footer">
                <p>🤖 This summary was generated by AI Video Summarizer</p>
                <p>For questions or feedback, please contact the meeting organizer</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def send_summary_email(self, recipients: List[str], 
                          summary_data: Dict[str, Any],
                          meeting_title: str = "Meeting Summary",
                          attachments: List[Path] = None) -> bool:
        """
        Send summary email to recipients.
        
        Args:
            recipients: List of recipient email addresses
            summary_data: Summary data dictionary
            meeting_title: Title of the meeting
            attachments: List of file paths to attach
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.sender_email or not self.sender_password:
            logger.error("Email credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Meeting Summary: {meeting_title}"
            
            # Create HTML content
            html_content = self.create_summary_email(summary_data, meeting_title, recipients)
            
            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Add attachments if provided
            if attachments:
                for attachment_path in attachments:
                    if attachment_path.exists():
                        with open(attachment_path, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {attachment_path.name}'
                            )
                            msg.attach(part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Summary email sent successfully to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False
    
    def send_bulk_emails(self, email_list: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Send bulk emails to multiple recipients with different content.
        
        Args:
            email_list: List of dictionaries containing:
                - recipients: List of email addresses
                - summary_data: Summary data
                - meeting_title: Meeting title
                - attachments: Optional list of file paths
        
        Returns:
            Dictionary mapping email addresses to success status
        """
        results = {}
        
        for email_data in email_list:
            recipients = email_data.get('recipients', [])
            summary_data = email_data.get('summary_data', {})
            meeting_title = email_data.get('meeting_title', 'Meeting Summary')
            attachments = email_data.get('attachments', [])
            
            success = self.send_summary_email(
                recipients, summary_data, meeting_title, attachments
            )
            
            for recipient in recipients:
                results[recipient] = success
        
        return results

def main():
    """Example usage of the EmailSender class."""
    # Example configuration
    sender = EmailSender(
        sender_email="your_email@gmail.com",
        sender_password="your_app_password"
    )
    
    # Example summary data
    summary_data = {
        "summary": "This is a sample meeting summary with key points discussed.",
        "action_items": ["Follow up on project proposal", "Schedule next meeting"],
        "keywords": ["project", "meeting", "proposal", "deadline"],
        "named_entities": [("John Doe", "PERSON"), ("Acme Corp", "ORG")],
        "metadata": {
            "original_sentence_count": 50,
            "summary_sentence_count": 5,
            "compression_ratio": 0.1,
            "action_item_count": 2,
            "keyword_count": 4,
            "named_entity_count": 2
        }
    }
    
    # Send email
    recipients = ["participant1@example.com", "participant2@example.com"]
    success = sender.send_summary_email(
        recipients=recipients,
        summary_data=summary_data,
        meeting_title="Weekly Team Meeting"
    )
    
    if success:
        print("Email sent successfully!")
    else:
        print("Failed to send email")

if __name__ == "__main__":
    main()
