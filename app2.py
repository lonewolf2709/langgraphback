import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, MessageGraph, END
from typing import TypedDict, Annotated, Sequence, Union, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import asyncio
import google.generativeai as genai
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
key = os.getenv("AZURE_SEARCH_API_KEY")
app = FastAPI()
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
workflow = StateGraph(AgentState)
search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
def send_email2(query):
    print("There might be an Error. An Email has been sent to the team and they will look into the issue")
    sender_email = "saad26ahmed@gmail.com"
    receiver_email = "l93186987@gmail.com"
    subject = query
    body = "The answer to the above query was not found"
    msg = MIMEMultipart()
    msg['From'] = "saad26ahmed@gmail.com"
    msg['To'] = "l93186987@gmail.com"
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = os.getenv('EMAIL_USER')
    smtp_password = os.getenv('EMAIL_PASSWORD')
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
        server.login(smtp_username, smtp_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        server.quit()
class QueryRequest(BaseModel):
    query: str
async def query_and_answer(state: AgentState) -> AgentState:
    query = state["query"]
    results = search_client.search(query, top=20)
    documents = [item['content'] for result in results.by_page() for item in result]
    concatenated_documents = ' '.join(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=350, length_function=len)
    chunks = text_splitter.split_text(text=concatenated_documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    docs = vector_store.similarity_search(query=query, k=18)
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    prompt_template = """
    You are an intelligent assistant. You are given a query and some context. Your task is to find the answer to the query within the provided context. If the answer is found, provide the answer directly. 
    If the answer cannot be found in the context, you must explicitly state 'send_email' and provide a reason.
    Context: {context}
    Query: {question}
    Answer the query or state 'send_email' if the answer is not found in the context.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    response = await chain.ainvoke({"context": docs, "question": query}, return_only_outputs=True)
    answer = response["text"]
    if "send_email" in answer:
        send_email2(query)
    new_messages = state["messages"] + [AIMessage(content=answer)]
    return {"messages": new_messages, "query": query}
workflow.add_node("chatbot", query_and_answer)
@app.post("/query/")
async def helper(requests: QueryRequest):
    query = requests.query
    if query:
        workflow.set_entry_point("chatbot")
        graph = workflow.compile()
        initial_state = {"messages": [HumanMessage(content=query)], "query": query}
        result = await graph.ainvoke(initial_state)
        return StreamingResponse(result_generator(result["messages"][-1].content), media_type="text/plain")
async def result_generator(answer):
    chunk_size = 200
    for i in range(0, len(answer), chunk_size):
        yield answer[i:i + chunk_size]
        await asyncio.sleep(0.3)
@app.get("/")
def read_root():
    return {"message": "Hello World"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)