"""StateGraph with a ReAct agent followed by a labeling agent."""

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from src.react_agent.graph import make_graph as make_react_graph
from src.labeling_agent.graph import make_graph as make_labeling_graph


async def make_two_node_graph(config: RunnableConfig):
    """Build a two node graph: ReAct agent -> labeling agent."""
    node1 = await make_react_graph(config)
    node2 = await make_labeling_graph(config)

    graph = StateGraph(dict)
    graph.add_node("node1", node1)
    graph.add_node("node2", node2)
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", END)
    graph.set_entry_point("node1")
    return graph.compile()
