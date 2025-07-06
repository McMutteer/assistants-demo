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
        You are a helpful AI assistant specialized in designing fun daily trivia bites!
        You have access to three tools: basic_research_tool, get_todays_date, and summary_report_tool.
        First, call get_todays_date, then perform any research if needed using basic_research_tool.
        Summarize your findings with summary_report_tool before producing a concise trivia fact.
        """

    # Compile the builder into an executable graph
    graph = create_react_agent(
        model=llm, 
        tools=tools, 
        prompt=prompt
    )

    return graph
