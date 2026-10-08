import smtplib
import mimetypes
from email.message import EmailMessage


def send_mail(recipient_email, Subject, body, sender_email, sender_password, attachments=None):
    """Send an email with optional file attachments.

    Args:
        recipient_email: Recipient's email address.
        Subject: Email subject line.
        body: Plain-text email body.
        sender_email: Gmail address used to send the email.
        sender_password: Gmail App Password for authentication.
        attachments: Optional list of Streamlit UploadedFile objects to attach.
    """
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = Subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    if attachments:
        for file in attachments:
            file_data = file.read()
            mime_type, _ = mimetypes.guess_type(file.name)
            if mime_type:
                main_type, sub_type = mime_type.split("/", 1)
            else:
                main_type, sub_type = "application", "octet-stream"
            msg.add_attachment(
                file_data,
                maintype=main_type,
                subtype=sub_type,
                filename=file.name,
            )

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {e}"