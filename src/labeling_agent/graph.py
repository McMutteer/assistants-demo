"""Labeling agent for extracting structured fields from text."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda

from src.utils import load_chat_model, get_message_text


async def make_graph(config: RunnableConfig):
    """Build a simple labeling agent.

    The returned runnable expects a text input and labels it with fields like
    ``Name``, ``Direcci\u00f3n``, ``Producto`` and ``Horario``.
    """
    configurable = config.get("configurable", {})
    model_name = configurable.get("model", "openai/gpt-4.1")
    system_prompt = configurable.get(
        "system_prompt",
        (
            "Eres un asistente que etiqueta el texto en los campos Name, Direcci\u00f3n, "
            "Producto y Horario. Devuelve el resultado en formato JSON."
        ),
    )

    llm = load_chat_model(model_name)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])

    chain = prompt | llm | RunnableLambda(get_message_text)
    return chain
