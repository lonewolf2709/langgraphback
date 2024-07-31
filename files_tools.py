import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import os
from  dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import PyPDF2
from docx import Document
from pptx import Presentation
import pandas as pd
from langchain.agents import initialize_agent, Tool
import json
import re
def convert_to_proper_json_string(json_str):
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", '"')
    
    # Ensure keys are properly quoted (handle nested structures)
    json_str = re.sub(r'(?<=\{|,)(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_str)
    
    return json_str

def get_summary(input):
    # Convert input string to proper JSON string
    proper_json_str = convert_to_proper_json_string(input)
    
    # Fix issues with document strings
    proper_json_str = proper_json_str.replace('Document(', '').replace(')', '')

    # Load the JSON string as a dictionary
    try:
        input_dict = json.loads(proper_json_str)
    except json.JSONDecodeError as e:
        print("Invalid JSON data:", e)
        return None

    # Extract the 'docs' and 'query' fields
    docs = input_dict.get('docs', [])
    query = input_dict.get('query', '')

    # Join the page contents of all documents
    docs_content = " ".join(doc.get('page_content', '') for doc in docs)

    # Define the model and template
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    generic_template = '''Write a summary of the following text: Text : `{speech}`.'''
    prompt = PromptTemplate(input_variables=['speech'], template=generic_template)
    llm_chain = LLMChain(llm=model, prompt=prompt)
    # Run the model with the extracted 'docs' content
    summary = llm_chain.run({'speech': docs_content})
    print(summary)
def query_finder(input):
    # Convert input string to proper JSON string
    proper_json_str = convert_to_proper_json_string(input)
    # Fix issues with document strings
    proper_json_str = proper_json_str.replace('Document(', '').replace(')', '')

    # Load the JSON string as a dictionary
    try:
        input_dict = json.loads(proper_json_str)
    except json.JSONDecodeError as e:
        print("Invalid JSON data:", e)
        return None

    # Extract the 'docs' and 'query' fields
    docs = input_dict.get('docs', [])
    query = input_dict.get('query', '')

    # Join the page contents of all documents
    context = " ".join(doc.get('page_content', '') for doc in docs)

    # Define the prompt template
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    print(context,query)
    # Initialize the model and the chain
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)

    try:
        response = chain.run({"context": context, "question": query})
    except Exception as e:
        # Handle parsing errors by passing it back to the agent
        print("Error:", e)
        return {"error": str(e), "retry": True}
    # Print and return the response
    print(response)
    return response
query_tool = Tool(
    name="Query_Solver",
    args={input},
    description="You are given a text and a query and this tool is used to answer to the query",
    func=query_finder
)
summary_tool=Tool(
    name="Summary_Creator",
    args={input},
    description="The text is given as an input and this tool is used to find the Summary of the text",
    func=get_summary
)
