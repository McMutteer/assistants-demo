from src.react_agent.tools import (
    basic_research_tool,
    get_todays_date,
    summary_report_tool,
)
from langgraph.prebuilt import create_react_agent
from src.utils import load_chat_model

async def make_graph():
    
    # initialize our model and tools    
    llm = load_chat_model("openai/gpt-4.1-mini")
    tools = [basic_research_tool, get_todays_date, summary_report_tool]
    prompt = """
        Eres un agente de una tienda que vende chocolates. Tu trabajo es tomar pedidos a domicilio de forma amable y clara. 
        Puedes usar la herramienta summary_report_tool si necesitas resumir detalles del pedido.
        """

    # Compile the builder into an executable graph
    graph = create_react_agent(
        model=llm, 
        tools=tools, 
        prompt=prompt
    )

    return graph
