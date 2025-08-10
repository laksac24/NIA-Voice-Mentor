# import fitz
# from typing import Annotated
# from typing_extensions import TypedDict
# from langgraph.graph.message import add_messages
# from langchain_core.prompts import ChatPromptTemplate

# def extract_text(pdf_path):
#     doc = fitz.open(pdf_path)
#     all_text = ""
#     for page in doc:
#         all_text += page.get_text()
#     doc.close()
#     return all_text

# text = extract_text("/Users/lakshyasachdeva/Documents/NIA/Lakshya_Resume.pdf")



# class State(TypedDict) :
#     messages:Annotated[list,add_messages]

# import os
# from dotenv import load_dotenv
# load_dotenv()

# from langchain_groq import ChatGroq
# llm=ChatGroq(model="Llama3-70b-8192")


# from langgraph.graph import StateGraph,START,END

# def resume_llm(state:State):
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", """
#                 You are an expert career mentor and advisor for students.

#                 Your job is to analyze the given extracted text from the user resume, and tell them:
#                 - A brief overview of their resume.
#                 - Which are their strong skills and which are their weak skills and where they have to focus on.
#                 - Future career opportunities and job roles.
#                 - How can they improve their resume.
#             THEE RESPONSE MUST BE HUMAN LIKE IN A VERY FRIENDLY TONE.

#                 Guidelines:
#                 - Dont add any ** or something to show bold letters and all
#                 - Keep the response short, NOT AI generated but well-structured, properly formatted and easy to read.
#                 - Use points where possible.
#                 - Use friendly but professional tone.
                
#                 Only give helpful, up-to-date and actionable guidance.

#                 If the user query is vague, ask a clarifying question to narrow it down.

#              """),
#              ("user", "command: {command}")
#         ]
#     )
#     chain = prompt|llm
#     return {"messages": [chain.invoke(state["messages"])]}

# # graph
# def guidance(query):
#     graph_builder = StateGraph(State)
#     graph_builder.add_node("resume_node",resume_llm)

#     # edges
#     graph_builder.add_edge(START,"resume_node")
    
#     graph_builder.add_edge("resume_node",END)

#     graph=graph_builder.compile()

#     response = graph.invoke({"messages":query})
#     return response["messages"][-1].content

# # web_answer("any ai discoveries today?")
# output = guidance(text)
# print(output)



# main.py
import fitz
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI, UploadFile, File
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# LangGraph State
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="Llama3-70b-8192")

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text()
    doc.close()
    return all_text

def resume_llm(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
                You are an expert career mentor and advisor for students.
                Your job is to analyze the given extracted text from the user resume, and tell them:
                - A brief overview of their resume.
                - Which are their strong skills and which are their weak skills and where they have to focus on.
                - Future career opportunities and job roles.
                - How can they improve their resume.
                THE RESPONSE MUST BE HUMAN LIKE IN A VERY FRIENDLY TONE.

                Guidelines:
                - Don't use ** or markdown formatting
                - Keep the response short but well-structured and easy to read
                - Use points where possible
                - Be friendly but professional
            """),
            ("user", "command: {command}")
        ]
    )
    chain = prompt | llm
    return {"messages": [chain.invoke(state["messages"])]}

def guidance(query):
    graph_builder = StateGraph(State)
    graph_builder.add_node("resume_node", resume_llm)
    graph_builder.add_edge(START, "resume_node")
    graph_builder.add_edge("resume_node", END)
    graph = graph_builder.compile()
    response = graph.invoke({"messages": query})
    return response["messages"][-1].content

# FastAPI app
app = FastAPI()

@app.post("/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    # Save the uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    text = extract_text(tmp_path)
    os.remove(tmp_path)
    result = guidance(text)
    return {"analysis": result}
