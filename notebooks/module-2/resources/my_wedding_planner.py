import asyncio
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage 
from typing import Dict, Any
from tavily import TavilyClient


load_dotenv()

#  -----------------------------------------***** FLIGHTS *****------------------------------------------------------------------

async def flights_search():
    client_flights= MultiServerMCPClient(
        {
  
    "kiwi-com-flight-search": {
      "url": "https://mcp.kiwi.com",
      "transport":"http",
                }
            
        }
    )
    tools = await client_flights.get_tools()
    
    flight_agent=create_agent(model="gpt-5-nano", 
                           tools= tools,
                           checkpointer=InMemorySaver(),)
    
    config = {"configurable": {"thread_id": "Plan"}}
    
    messages_F=[SystemMessage(content="You are a travel agent and will help the user search flights to and from the selected destination. Just pick two options, one for each direction. No follow up questions."),
                HumanMessage(content="I want to celebrate my wedding at Toronto. Can you find me some flights from Madrid to Toronto?")]
    
    response = await flight_agent.ainvoke({"messages":messages_F},
                                       config)
    
    print(response["messages"][-1].content)
    
asyncio.run(flights_search())

#--------------------------------------------*****VENUE*****-------------------------------------------------------------------------------------------------------

tavily_client=TavilyClient()

@tool
def web_search(query:str) -> Dict[str,Any]:
    """Search the web for a information"""
    return tavily_client.search(query)


    
    


    

