"""Run the arXiv MCP agent from a plain Python script (asyncio).

Same flow as resources/mine.ipynb, but as a standalone script:
1. Load env vars (OpenAI key, etc.) from .env
2. Connect to the arxiv MCP server over stdio
3. Fetch its tools and build a LangChain agent
4. Ask a question and print the answer

Why asyncio: the MCP client talks to the server over an async transport, so
`get_tools()` and the agent's `ainvoke()` are coroutines. In a notebook there is
already a running event loop so you can `await` at the top level; in a plain
script there is none, so we wrap everything in `main()` and start a loop with
`asyncio.run()`.
"""

import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

async def arxiv_mcp():
# Description of every MCP server the agent can reach. Here just one: a local
# arxiv server that we launch ourselves as a subprocess.
  MCP_CONFIG = {
      "arxiv-server": {
          # "stdio" = run the server as a child process and talk to it over
          # stdin/stdout (vs. "streamable_http" for a server reachable by URL).
          "transport": "stdio",
          # The command + args below are literally the process that gets spawned:
          #   uv --directory <server src> run server.py
          # `uv run` resolves that project's own dependencies before starting it.
          "command": "uv",
          "args": [
              "--directory",
              "/home/arlet/lca-lc-foundations/notebooks/module-2/arxiv-mcp-server/src/arxiv_server",
              "run",
              "server.py",
          ],
          # Extra environment variables passed to that subprocess. The arxiv
          # server reads DOWNLOAD_PATH to decide where to save downloaded PDFs.
          "env": {
              "DOWNLOAD_PATH": "/home/arlet/lca-lc-foundations/notebooks/module-2/arxiv-mcp-server/download_papers",
              "FASTMCP_LOG_LEVEL": "WARNING",
          },
      }
  }

  client = MultiServerMCPClient(MCP_CONFIG)


  tools= await client.get_tools()
  
  agent = create_agent(model="gpt-5-nano", tools=tools)

  question = HumanMessage(content="find me a paper about 6g")
  response = await agent.ainvoke({"messages": [question]})
  print(response["messages"][-1].content)
  
#ayncio.run(arxiv_mcp())

async def time_zoneConv():
    time_mcp= MultiServerMCPClient(
        {
  
    "kiwi-com-flight-search": {
      "url": "https://mcp.kiwi.com",
      "transport":"http",
                }
            
        }
    )
    tools = await time_mcp.get_tools()
    
    agentTime=create_agent(model="gpt-5-nano", 
                           tools= tools,
                           checkpointer=InMemorySaver(),)
    
    config = {"configurable": {"thread_id": "viajes"}}
    
    question=HumanMessage(content="buscame vuelos de Madrid a la praga ida y vuelta. entre el 5 y el 8 de diciembre, fechas flexibles. 1 pasajero adulto, M. no me hagas preguntas.")
    
    response = await agentTime.ainvoke({"messages":[question]},
                                       config)
    
    print(response["messages"][-1].content)
    
asyncio.run(time_zoneConv())
    
     