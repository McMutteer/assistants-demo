"""Define a Reasoning and Action agent using the LangGraph prebuilt react agent. 

Add configuration and implement using a make_graph function to rebuild the graph at runtime.
"""
from src.react_agent.tools import get_tools
from langgraph.prebuilt import create_react_agent
from src.utils import load_chat_model

from src.react_agent.configuration import Configuration
from langchain_core.runnables import RunnableConfig



async def make_graph(config: RunnableConfig):
    
    # Get name from config or use default
    configurable = config.get("configurable", {})

    # get values from configuration
    llm = configurable.get("model", "openai/gpt-4.1")
    selected_tools = configurable.get(
        "selected_tools", ["get_todays_date", "summary_report_tool"]
    )
    prompt = configurable.get(
        "system_prompt",
        (
            "Eres un asistente que atiende un negocio. "
            "Cuando completes un pedido, utiliza la herramienta summary_report_tool "
            "para generar un breve informe resumido. Responde siempre en español."
        ),
    )
    
    # specify the name for use in supervisor architecture
    name = configurable.get("name", "react_agent")

    # Compile the builder into an executable graph
    # You can customize this by adding interrupt points for state updates
    graph = create_react_agent(
        model=load_chat_model(llm), 
        tools=get_tools(selected_tools),
        prompt=prompt, 
        config_schema=Configuration,
        name=name
    )

    return graph