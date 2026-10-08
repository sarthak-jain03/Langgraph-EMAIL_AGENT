# Email Agent — LangGraph + Streamlit

An agentic email assistant built with LangGraph and Streamlit. You describe what you want to send and to whom — the AI drafts it, you review it, tweak it if needed, and send it directly from your Gmail.

**Live demo → [email-agent-01.streamlit.app](https://email-agent-01.streamlit.app/)**

---

## What it does

- Pulls the recipient details (name, email) and intent out of a plain-English description
- Drafts a properly formatted email using an LLM
- Lets you review the draft before anything gets sent
- You can ask for revisions as many times as you want
- Attach files (PDF, DOCX, images, etc.) — they go along with the email
- If you attach a document like a resume or a job posting, the LLM reads it and writes the email around it

---

## How the agent works

The graph has five nodes wired together with LangGraph:

```
START → retriever → draft → review ⟶ send → END
                              ↑         |
                              └── draft ┘  (on feedback)
```

- **retriever** — extracts recipient name, email, and the reason for the email from your input (and any attached documents)
- **draft** — writes the email subject and body; on revision loops it rewrites based on your feedback
- **review** — pauses the graph and waits for your approval or feedback (`interrupt`)
- **send** — fires off the email via Gmail SMTP with any attachments

State is persisted across the interrupt using `InMemorySaver`, so the graph can be resumed after you approve or give feedback.

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/sarthak-jain03/Langgraph-EMAIL_AGENT.git
cd Langgraph-EMAIL_AGENT
pip install -r requirements.txt
```

**2. Get a Groq API key**

Sign up at [console.groq.com](https://console.groq.com) and grab a free API key.

**3. Create a `.env` file**

```
GROQ_API_KEY=your_key_here
```

**4. Get a Gmail App Password**

Your regular Gmail password won't work here — you need an App Password:

1. Go to your [Google Account](https://myaccount.google.com/) → Security
2. Enable 2-Step Verification if you haven't already
3. Search for "App passwords" and generate one for Mail
4. Copy the 16-character password (spaces are fine)

**5. Run it**

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), enter your Gmail address and App Password in the sidebar, and you're good to go.

---

## Attaching documents

When you attach a file in the compose screen, the agent extracts the text before drafting. This means:

- Attach your **resume** when applying for a job — the LLM will pull relevant skills and experience into the email automatically
- Attach a **job posting** — it'll pick up the recruiter's email and tailor the message to the role
- Attach any reference document you want the email to reference

Supported formats: PDF, DOCX, TXT, MD, and most other text-based files.

The same file is also sent as an attachment in the actual email.

---

## Project structure

```
Email-Agent/
├── app.py                 # Streamlit UI + LangGraph agent
├── agent.py               # Standalone CLI version of the agent
├── send_email.py          # Gmail SMTP sender with attachment support
├── document_extractor.py  # Text extraction from PDF / DOCX / TXT
├── requirements.txt
└── .env                   # Your API key 
```

---

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent graph with human-in-the-loop via `interrupt`
- [LangChain Groq](https://python.langchain.com/docs/integrations/chat/groq/) — LLM inference
- [Streamlit](https://streamlit.io) — UI
- `smtplib` — email sending (standard library, no extra deps)
- `pypdf` + `python-docx` — document text extraction

---

## Notes

- Credentials are never stored anywhere. They live in your browser session only and are used solely to authenticate with Gmail's SMTP server.
- The agent runs entirely locally. No emails or documents are sent to any external service other than Groq for inference and Gmail for sending.
- The CLI version (`agent.py`) works without Streamlit if you want to run it in a terminal.
