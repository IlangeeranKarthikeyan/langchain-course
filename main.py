
from dotenv import load_dotenv

load_dotenv()
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")
    name: str = Field(description="The name of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The sources used to generate the response")

tavily_client = TavilyClient()

# Tool call using tavily client
# @tool 
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The search query
#     Returns:
#         The search results
#     """
#     print(f"Searching for: {query}")
#     return tavily_client.search(query)

llm = ChatOpenAI(model="gpt-5", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Agent is running...")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job openings for an ai engineer using langchain in bay area on linkedin and list their details")]})
    print(result)

if __name__ == "__main__":
    main()