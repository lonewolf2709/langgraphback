from langchain.agents import initialize_agent, load_tools, AgentType, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from files_tools import query_tool,summary_tool
model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0)
tools = [query_tool,summary_tool]
agent = initialize_agent(tools=tools,llm=model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
