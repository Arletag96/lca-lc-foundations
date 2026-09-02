import asyncio
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage 
from typing import Dict, Any
from tavily import TavilyClient
from langchain_community.utilities import SQLDatabase


load_dotenv()

#  -----------------------------------------***** FLIGHTS *****------------------------------------------------------------------

async def flights_search():     #to modify----*/*/*/*/*/*/*/*/*
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
    """Search the web to look for a perfect venue to host the wedding event"""
    return tavily_client.search(query)

venue_agent= create_agent(
    model="gpt-5-nano",
    tools=web_search,
)


#----------------------------------------*****PAYLIST*****---------------------------------------------------------------------------------------   
    
db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")
@tool
def SQL_query(query: str) -> str:

    """Obtain information from the database using SQL queries"""

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"

playlist_agent = create_agent(
    model="gpt-5-nano",
    tools=[SQL_query]
)

#----------------------------------*****COORDINATOR's TOOLS (SUBAGENTS)*****-------------------------------------------------------------------------------------------    

@tool
def call_flight_agent():
    """Call flight agent in order to look for flights between two locations (source and destiny) in two specificated dates"""
    response_F= flight_agent.invoke({"messages":[SystemMessage(content="You are a travel agent and will help the user search flights to and from the selected destination. Just pick two options, one for each direction. No follow up questions."),
                HumanMessage(content="I want to celebrate my wedding at Toronto. Can you find me some flights from Madrid to Toronto?")]})
    return response_F["messages"][-1].content
@tool
def call_venue_agent():
    """Call venue agent to look in the web for an appropriate venue to host a wedding event"""
    response_V=venue_agent.invoke({"messages":[HumanMessage(content=f"You are a planner agent that looks for a perfect venue to host a wedding event in {location}")]})
    return response_V["messages"][-1].content

@tool
def call_playlist_agent():
    """Call plylist agent to look in a db for a playlist for a wedding celebration"""
    response_PL=playlist_agent.invoke({"messages":[HumanMessage(content=f"You are dj agent that looks for a perfect playlist for a wedding in {location}")]})
    return response_PL["messages"][-1].content


#------------------------------*****COORDINATOR AGENT*****-------------------------------------------------------------------------------------------------

coord_agent=create_agent(
    model="gpt-5-nano",
    tools=[call_flight_agent,call_venue_agent,call_playlist_agent],
    system_prompt="You are a helpful assistant who can call some other agents to organize a wedding"
)

question=""
response=coord_agent.invoke({"messages":[HumanMessage(content=question)]})