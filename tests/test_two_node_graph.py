import pytest
from langchain_core.runnables import RunnableConfig
from src.two_node_graph.graph import make_two_node_graph

@pytest.mark.asyncio
async def test_two_node_graph_ainvoke():
    config = RunnableConfig(configurable={"model": "openai/gpt-4"})
    graph = await make_two_node_graph(config)
    await graph.ainvoke({"input": "test"})
