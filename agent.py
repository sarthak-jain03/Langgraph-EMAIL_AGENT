from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing import Literal
from send_email import send_mail
import os

os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-20b")

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



def retriever_node(state: EmailState) -> EmailState:
    llm_for_userdetails = llm.with_structured_output(UserDetails)
    prompt = (
        "Extract the user details for sending an email. "
        "You MUST respond by calling the provided tool to output the structured data. "
        "If the query is empty or missing information, output empty strings for those fields.\n\n"
        f"Query: {state.question}"
    )
    userDetails:UserDetails = llm_for_userdetails.invoke(prompt)

    if not userDetails.mail_reason or not userDetails.recipient_email:
        pass

    state.recipient_name = userDetails.recipient_name
    state.recipient_email = userDetails.recipient_email
    state.mail_reason = userDetails.mail_reason

    return state




class DraftEmail(BaseModel):
    subject: str = Field(description="Email subject within 40 words")
    body: str = Field(description="Email body including proper detail, reason and greetings and others")



def draft_node(state: EmailState) -> EmailState:
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
        {
            "message": "You need to approve this or re-write this email."
        }
    )

    if response == "yes":
        state.feedback = ''
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
    "Send Final Email"
    res = send_mail(state.recipient_email, state.subject, state.body)
    state.response = res
    return state




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


final_graph = graph.compile(checkpointer=InMemorySaver())




if __name__ == "__main__":
    while True:
        query = input("User: ")
        if query == "quit":
            print("Bye 👋")
            break
        
        config = {"configurable": {"thread_id":"1"}}
        res = final_graph.invoke(
            {"question":query},
            config=config
        )

        while True:
            state = final_graph.get_state(config)
            
            if not state.next:
                break
            
            print("\n", "-"*60)
            print("Subject: ", res.get("subject", ""))
            print("Body: ", res.get("body", ""))
            print("\n", "-"*60)
            

            feedback = input("Approve to send the mail or provide the feedback: ")
            res = final_graph.invoke(
                    Command(resume=feedback),
                    config=config
                )         
        
        print("AI: ", res.get("response", ""))


        

        
        

    
    