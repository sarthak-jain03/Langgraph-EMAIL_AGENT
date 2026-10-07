import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv()
import os


def send_mail(recipient_email, Subject, body):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")


    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = Subject
    msg["From"]  = sender_email
    msg["To"] = recipient_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()  # Upgrade the insecure connection to TLS
            server.ehlo()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {e}"