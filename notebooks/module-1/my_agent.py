from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain.agents import create_agent
from pprint import pprint
from langgraph.checkpoint.memory import InMemorySaver  
from pydantic import BaseModel
from ipywidgets import FileUpload
from IPython.display import display
import base64
from langchain_core.callbacks import StdOutCallbackHandler

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)
  
model = init_chat_model(
    model="gpt-5-nano",
    # Kwargs passed to the model:
    temperature=0.5,
    timeout=60
)
system_prompt=""" you are a personal cheff. The user will give you a picture of the food he has in his refrigerator and you will search online (using the web search tool)
for a recipe with those aliments as ingredients. Then give to the user some recipe suggestions and a few steps for its preparation. Give the url as well.
    """

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)
#memory
#config = {"configurable": {"thread_id": "1"}}

path = "/home/arlet/foto/49201404-domestic-refrigerator-stocked-with-fresh-and-healthy-food.png"

with open(path, "rb") as f:
    img_bytes = f.read()

img_b64 = base64.b64encode(img_bytes).decode("utf-8")

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "I have some left over chicken and rice. What can I make adding what i have in my fridge?"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

"""response = agent.invoke(
    {"messages": [multimodal_question]}
)"""

response = agent.invoke(
    {"messages": [multimodal_question]},
    config={#"callbacks": [StdOutCallbackHandler()], 
            "configurable": {"thread_id": "1"}}
)

print(response["messages"][-1].content)