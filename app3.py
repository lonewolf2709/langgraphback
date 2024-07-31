from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langgraph.prebuilt import ToolExecutor, ToolNode
from langgraph.graph import StateGraph, MessageGraph, END
from common.utils import DocSearchAgent, BingSearchAgent
from callbacks import StdOutCallbackHandler
from dotenv import load_dotenv
from typing import List, Dict, Any
from prompts import CUSTOM_CHATBOT_PROMPT, CUSTOM_CHATBOT_PREFIX
from typing import TypedDict, Annotated, Sequence, Union, List
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
import operator
import uvicorn
from fastapi.responses import StreamingResponse
import asyncio
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Message(BaseModel):
    role: str
    content: str
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
cb_handler = StdOutCallbackHandler()
cb_manager = CallbackManager(handlers=[cb_handler])
COMPLETION_TOKENS = 2000
llm = AzureChatOpenAI(
    deployment_name=os.getenv("GPT35_DEPLOYMENT_NAME"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.5, 
    max_tokens=COMPLETION_TOKENS, 
    streaming=True, 
    callback_manager=cb_manager
)
doc_indexes=[]
doc_search = DocSearchAgent(llm=llm, indexes=doc_indexes, k=6, reranker_th=1, name="docsearch", description="useful when the questions includes the term: docsearch", verbose=True)
www_search = BingSearchAgent(llm=llm, k=1, name="bing", description="useful when the questions includes the term: bing", verbose=True)
tools=[doc_search, www_search]
llm_with_tools = llm.bind_tools(tools)
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        arguments = last_message.tool_calls[0]["args"]
        if arguments.get("return_direct", False):
            return "final"
        else:
            return "continue"
def supervisor_node(state):
    messages = state["messages"]
    PROMPT = ChatPromptTemplate.from_messages(
        [("system", CUSTOM_CHATBOT_PREFIX), ("human", "{question}")]
    )
    chain = ({"question": lambda x: x["question"]} | PROMPT | llm_with_tools)
    response = chain.invoke({"question": messages})
    return {"messages": [response]}
tool_node = ToolNode(tools)
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("tools", tool_node)
workflow.add_node("tools_final", tool_node)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    should_continue,
    {
        "continue": "tools",
        "final": "tools_final",
        "end": END,
    },
)
workflow.add_edge("tools", "supervisor")
workflow.add_edge("tools_final", END)
memory = AsyncSqliteSaver.from_conn_string(":memory:")
app_workflow = workflow.compile(checkpointer=memory)
class QueryRequest(BaseModel):
    query: str
@app.post("/query")
async def invoke_workflow(requests:QueryRequest):
    query = requests.query
    print(query)
    initial_state = {"messages": [HumanMessage(content=query)], "query": query}
    async def result_generator():
        result = await app_workflow.ainvoke(initial_state, {"configurable": {"thread_id": "3"}})
        answer = result["messages"][-1].content
        chunk_size = 200
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]
            await asyncio.sleep(0.5)
    return StreamingResponse(result_generator(), media_type="text/plain")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)