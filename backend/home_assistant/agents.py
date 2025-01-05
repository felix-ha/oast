from typing import Annotated
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END


model = "llama3.2:3b"
llm = ChatOllama(model=model,
                 base_url="http://localhost:11434")


class State(BaseModel):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    result = llm.invoke(state.messages)
    return {"messages": [result]}

def get_graph():
    memory = MemorySaver()
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile(checkpointer=memory)
    return graph


if __name__ == "__main__":
    graph = get_graph()
    config = {"configurable": {"thread_id": "1"}}

    user_input = "hi, my name is felix"
    result_1 = graph.invoke(input={"messages": [("user", user_input)]}, config=config)
    print(result_1)

