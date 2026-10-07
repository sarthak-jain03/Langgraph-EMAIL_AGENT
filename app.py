import streamlit as st
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing import Literal
from send_email import send_mail


st.set_page_config(
    page_title="Email Agent – AI-Powered Email Sender",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .hero {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #d97757 0%, #e8a365 50%, #f4c095 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero p {
        color: #7a6a5a;
        font-size: 1.05rem;
        font-weight: 400;
    }

    .email-card {
        background: linear-gradient(145deg, #ffffff 0%, #f9f5f0 100%);
        border: 1px solid rgba(217, 119, 87, 0.3);
        border-radius: 16px;
        padding: 1.75rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(217, 119, 87, 0.08);
    }
    .email-card .label {
        color: #d97757;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }
    .email-card .value {
        color: #4a4036;
        font-size: 0.95rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .email-card .subject-value {
        color: #2c251f;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .email-card hr {
        border: none;
        border-top: 1px solid rgba(217, 119, 87, 0.15);
        margin: 0.75rem 0;
    }

    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-success {
        background: rgba(101, 163, 101, 0.15);
        color: #487e48;
        border: 1px solid rgba(101, 163, 101, 0.3);
    }
    .status-error {
        background: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .status-info {
        background: rgba(217, 119, 87, 0.15);
        color: #c46243;
        border: 1px solid rgba(217, 119, 87, 0.3);
    }
    .status-warning {
        background: rgba(234, 179, 8, 0.15);
        color: #ca8a04;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 1.5rem 0;
    }
    .step {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.8rem;
        color: #a39587;
    }
    .step.active {
        color: #d97757;
        font-weight: 600;
    }
    .step.done {
        color: #65a365;
    }
    .step-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #e2d7c9;
    }
    .step.active .step-dot {
        background: #d97757;
        box-shadow: 0 0 8px rgba(217, 119, 87, 0.5);
    }
    .step.done .step-dot {
        background: #65a365;
    }
    .step-line {
        width: 24px;
        height: 2px;
        background: #e2d7c9;
        align-self: center;
    }

    .cred-card {
        background: linear-gradient(145deg, #ffffff 0%, #f9f5f0 100%);
        border: 1px solid rgba(217, 119, 87, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }
    .cred-card h4 {
        color: #2c251f;
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
    }
    .cred-card p {
        color: #7a6a5a;
        font-size: 0.8rem;
        line-height: 1.5;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)



def get_groq_api_key():
    """Get GROQ API key from st.secrets (deployment) or .env (local)."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY", "")



@st.cache_resource
def get_llm():
    api_key = get_groq_api_key()
    return ChatGroq(model="openai/gpt-oss-20b", api_key=api_key)


class EmailState(BaseModel):
    question: str = ""
    mail_reason: str = ""
    recipient_name: str = ""
    recipient_email: str = ""
    subject: str = ""
    body: str = ""
    feedback: str = ""
    feedback_count: int = 0
    response: str = ""


class UserDetails(BaseModel):
    mail_reason: str = Field(description="Reason of the email, like why the user want to send this email")
    recipient_name: str = Field(description="Recipient / reciever name, if available, else '' ")
    recipient_email: str = Field(description="Recipient / reciever's email address")


class DraftEmail(BaseModel):
    subject: str = Field(description="Email subject within 40 words")
    body: str = Field(description="Email body including proper detail, reason and greetings and others")


def retriever_node(state: EmailState) -> EmailState:
    llm = get_llm()
    llm_for_userdetails = llm.with_structured_output(UserDetails)
    prompt = (
        "Extract the user details for sending an email. "
        "You MUST respond by calling the provided tool to output the structured data. "
        "If the query is empty or missing information, output empty strings for those fields.\n\n"
        f"Query: {state.question}"
    )
    userDetails: UserDetails = llm_for_userdetails.invoke(prompt)
    state.recipient_name = userDetails.recipient_name
    state.recipient_email = userDetails.recipient_email
    state.mail_reason = userDetails.mail_reason
    return state


def draft_node(state: EmailState) -> EmailState:
    llm = get_llm()
    draft_email_llm = llm.with_structured_output(DraftEmail)

    if state.feedback:
        prompt = f"""
            Revise this email based on the feedback below:
            Current email: 
            Subject: {state.subject}
            Body: {state.body}
            Feedback: {state.feedback}

            Please write a proper email body and subject without extra text and improvement based on the feedback.
            The max size of the mail body should be 200 words.
        """
    else:
        prompt = f"""
            Write an email with these details:
            To: {state.recipient_name}
            Request: {state.mail_reason}

            Please write a proper mail body and subject without extra text.
            The maximum size of the mail body is 200 words.
        """

    draft_email: DraftEmail = draft_email_llm.invoke(prompt)
    state.subject = draft_email.subject
    state.body = draft_email.body
    return state


def review_node(state: EmailState) -> EmailState:
    response = interrupt(
        {"message": "You need to approve this or re-write this email."}
    )
    if response == "yes":
        state.feedback = ""
    else:
        state.feedback = response
        state.feedback_count = state.feedback_count + 1
    return state


def router(state: EmailState) -> Literal["draft", "send", "cancel"]:
    if not state.feedback or state.feedback.strip() == "":
        return "send"
    if state.feedback_count > 2:
        return "cancel"
    return "draft"


def cancel_node(state: EmailState) -> EmailState:
    state.response = "Email not sent. You have already reached the maximum limit of feedback!"
    return state


def send_node(state: EmailState) -> EmailState:
    """Send Final Email using the user's credentials from session state."""
    sender_email = st.session_state.get("sender_email", "")
    sender_password = st.session_state.get("sender_password", "")
    res = send_mail(state.recipient_email, state.subject, state.body, sender_email, sender_password)
    state.response = res
    return state


@st.cache_resource
def build_graph():
    graph = StateGraph(EmailState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("draft", draft_node)
    graph.add_node("review", review_node)
    graph.add_node("cancel", cancel_node)
    graph.add_node("send", send_node)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "draft")
    graph.add_edge("draft", "review")
    graph.add_conditional_edges("review", router)
    graph.add_edge("send", END)
    graph.add_edge("cancel", END)

    return graph.compile(checkpointer=InMemorySaver())


final_graph = build_graph()



if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "stage" not in st.session_state:
    st.session_state.stage = "input"

if "current_draft" not in st.session_state:
    st.session_state.current_draft = None

if "final_result" not in st.session_state:
    st.session_state.final_result = None

if "extracted_details" not in st.session_state:
    st.session_state.extracted_details = None

if "credentials_saved" not in st.session_state:
    st.session_state.credentials_saved = False


def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def reset_session():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.stage = "input"
    st.session_state.current_draft = None
    st.session_state.final_result = None
    st.session_state.extracted_details = None



def render_steps(current):
    steps = ["Connect", "Compose", "Review", "Send"]
    idx_map = {"credentials": 0, "input": 1, "review": 2, "done": 3}
    current_idx = idx_map.get(current, 0)

    html = '<div class="step-indicator">'
    for i, s in enumerate(steps):
        if i < current_idx:
            cls = "step done"
        elif i == current_idx:
            cls = "step active"
        else:
            cls = "step"
        html += f'<div class="{cls}"><span class="step-dot"></span>{s}</div>'
        if i < len(steps) - 1:
            html += '<div class="step-line"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)



st.markdown("""
<div class="hero">
    <h1>Email Agent</h1>
    <p>AI-powered email drafting & sending with human-in-the-loop review</p>
</div>
""", unsafe_allow_html=True)



with st.sidebar:
    st.markdown("## 🔐 Your Gmail Credentials")
    st.markdown(
        '<p style="color:#94a3b8;font-size:0.82rem;">'
        "Your credentials are used only to send emails and are never stored on any server."
        "</p>",
        unsafe_allow_html=True,
    )

    sender_email = st.text_input(
        "Gmail Address",
        value=st.session_state.get("sender_email", ""),
        placeholder="you@gmail.com",
        key="sidebar_email",
    )
    sender_password = st.text_input(
        "App Password",
        value=st.session_state.get("sender_password", ""),
        type="password",
        placeholder="xxxx xxxx xxxx xxxx",
        key="sidebar_password",
    )

    if st.button("💾 Save Credentials", use_container_width=True):
        if sender_email and sender_password:
            st.session_state.sender_email = sender_email
            st.session_state.sender_password = sender_password
            st.session_state.credentials_saved = True
            st.success("✅ Credentials saved for this session!")
        else:
            st.error("Please fill in both fields.")

    if st.session_state.credentials_saved:
        st.markdown(
            f'<span class="status-badge status-success">✅ Connected as {st.session_state.sender_email}</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    with st.expander("🔑 How to get a Gmail App Password"):
        st.markdown("""
        1. Go to [Google Account Settings](https://myaccount.google.com/)
        2. Navigate to **Security** → **2-Step Verification** (must be enabled)
        3. Scroll down and click **App passwords**
        4. Select **Mail** and your device
        5. Click **Generate** — copy the 16-character password
        6. Paste it above (spaces are fine)
        
        > ⚠️ This is **NOT** your regular Gmail password.
        """)

    st.divider()

    st.markdown("### How it works")
    st.markdown("""
    1. **Connect** – Enter your Gmail & App Password
    2. **Compose** – Describe your email in natural language
    3. **Review** – Approve or revise the AI draft (up to 3×)
    4. **Send** – Email is sent from your Gmail account
    """)

    st.divider()
    st.markdown(
        '<p style="color:#64748b;font-size:0.8rem;">Built with LangGraph + Streamlit<br>'
        '🔒 Credentials stay in your browser session only</p>',
        unsafe_allow_html=True,
    )



if not st.session_state.credentials_saved:
    render_steps("credentials")

    st.markdown(
        """
        <div class="cred-card">
            <h4>👋 Welcome! Let's get started.</h4>
            <p>
                To send emails, you need to connect your <strong>Gmail account</strong>
                using an <strong>App Password</strong>.<br><br>
                👈 Enter your credentials in the sidebar to begin.<br><br>
                Your credentials are <strong>never stored</strong> — they only exist in your
                current browser session and are used solely to send emails on your behalf.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()



render_steps(st.session_state.stage)


if st.session_state.stage == "input":
    st.markdown("### 📝 Compose Your Email")
    st.markdown(
        '<span style="color:#94a3b8;font-size:0.9rem;">'
        "Describe what you want to email and to whom. The AI will extract the details and draft it for you."
        "</span>",
        unsafe_allow_html=True,
    )

    with st.form("email_form", clear_on_submit=True):
        query = st.text_area(
            "Your request",
            placeholder='e.g. "Send an email to John (john@example.com) asking for a meeting next Monday at 3 PM"',
            height=120,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🚀 Generate Draft", use_container_width=True)

    if submitted and query.strip():
        with st.spinner("🤖 Extracting details & drafting your email..."):
            config = get_config()
            res = final_graph.invoke({"question": query}, config=config)

        state = final_graph.get_state(config)
        if state.next:
            st.session_state.current_draft = {
                "to": res.get("recipient_email", ""),
                "name": res.get("recipient_name", ""),
                "subject": res.get("subject", ""),
                "body": res.get("body", ""),
                "reason": res.get("mail_reason", ""),
            }
            st.session_state.extracted_details = {
                "to": res.get("recipient_email", ""),
                "name": res.get("recipient_name", ""),
                "reason": res.get("mail_reason", ""),
            }
            st.session_state.stage = "review"
            st.rerun()
        else:
            st.session_state.final_result = res.get("response", "Done!")
            st.session_state.stage = "done"
            st.rerun()



elif st.session_state.stage == "review":
    draft = st.session_state.current_draft
    details = st.session_state.extracted_details

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<span class="status-badge status-info">📧 {details["to"]}</span>',
            unsafe_allow_html=True,
        )
    with col2:
        if details["name"]:
            st.markdown(
                f'<span class="status-badge status-info">👤 {details["name"]}</span>',
                unsafe_allow_html=True,
            )
    with col3:
        feedback_count = final_graph.get_state(get_config()).values.get("feedback_count", 0)
        remaining = max(0, 3 - feedback_count)
        st.markdown(
            f'<span class="status-badge status-warning">🔄 {remaining} revisions left</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<span class="status-badge status-success">📤 Sending from: {st.session_state.sender_email}</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="email-card">
            <div class="label">From</div>
            <div class="value">{st.session_state.sender_email}</div>
            <hr>
            <div class="label">To</div>
            <div class="value">{draft["name"]} &lt;{draft["to"]}&gt;</div>
            <hr>
            <div class="label">Subject</div>
            <div class="subject-value">{draft["subject"]}</div>
            <hr>
            <div class="label">Body</div>
            <div class="value">{draft["body"].replace(chr(10), "<br>")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ✅ Review & Approve")

    col_approve, col_revise = st.columns(2)

    with col_approve:
        if st.button("✅ Approve & Send", use_container_width=True, type="primary"):
            with st.spinner("📤 Sending email..."):
                config = get_config()
                res = final_graph.invoke(Command(resume="yes"), config=config)
            st.session_state.final_result = res.get("response", "")
            st.session_state.stage = "done"
            st.rerun()

    with col_revise:
        pass

    with st.form("feedback_form", clear_on_submit=True):
        feedback = st.text_area(
            "Provide feedback to revise the email",
            placeholder='e.g. "Make it more formal" or "Add a greeting with Dear Sir/Madam"',
            height=80,
            label_visibility="collapsed",
        )
        revise_submitted = st.form_submit_button("🔄 Revise Draft", use_container_width=True)

    if revise_submitted and feedback.strip():
        with st.spinner("✏️ Revising your email..."):
            config = get_config()
            res = final_graph.invoke(Command(resume=feedback), config=config)

        state = final_graph.get_state(config)
        if state.next:
            st.session_state.current_draft = {
                "to": res.get("recipient_email", draft["to"]),
                "name": res.get("recipient_name", draft["name"]),
                "subject": res.get("subject", ""),
                "body": res.get("body", ""),
                "reason": res.get("mail_reason", draft.get("reason", "")),
            }
            st.rerun()
        else:
            st.session_state.final_result = res.get("response", "")
            st.session_state.stage = "done"
            st.rerun()



elif st.session_state.stage == "done":
    result = st.session_state.final_result or ""

    if "successfully" in result.lower():
        st.balloons()
        st.markdown(
            f"""
            <div style="text-align:center; padding: 2rem 0;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🎉</div>
                <h2 style="color:#4ade80; margin-bottom:0.5rem;">Email Sent!</h2>
                <span class="status-badge status-success">{result}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 2rem 0;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">⚠️</div>
                <h2 style="color:#facc15; margin-bottom:0.5rem;">Process Complete</h2>
                <span class="status-badge status-warning">{result}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    if st.button("📝 Compose Another Email", use_container_width=True, type="primary"):
        reset_session()
        st.rerun()
