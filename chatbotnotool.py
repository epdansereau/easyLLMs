import fire
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

import gradio as gr
from gradio_chatbot_UI import launch_gradio_chatbot

from models import get_model, load_settings

class State(TypedDict):
    messages: Annotated[list, add_messages]

def build_graph(model):
    def chatbot(state: State):
        return {"messages": [model.invoke(state["messages"])]}
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.set_entry_point("chatbot")
    graph_builder.set_finish_point("chatbot")
    return graph_builder

def stream_graph_updates_no_mem_no_stream(user_input: str, graph, system=None):
    if system:
        history = [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
    else:
        history = [{"role": "user", "content": user_input}]
    events = graph.stream({"messages": history})
    return events

def stream_graph_updates_no_stream(user_input: str, graph, config, system=None):
    if system:
        history = [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
    else:
        history = [{"role": "user", "content": user_input}]
    events = graph.stream(
        {"messages": history},
        config,
    )
    return events

def stream_graph_updates_no_mem(user_input: str, graph, system=None):
    if system:
        history = [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
    else:
        history = [{"role": "user", "content": user_input}]
    events = graph.stream({"messages": history}, stream_mode="messages")
    return events

def stream_graph_updates(user_input: str, graph, config, system=None):
    if system:
        history = [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
    else:
        history = [{"role": "user", "content": user_input}]
    events = graph.stream(
        {"messages": history},
        config,
        stream_mode="messages"
    )
    return events

def display_response(events):
    for event in events:
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

def stream_response(events):
    print("Assistant: ", end="", flush=True)
    for message_chunk, metadata in events:
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    print()  

def run_cli_chatbot_no_mem_no_stream(model):
    graph_builder = build_graph(model)
    graph = graph_builder.compile()
    while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            events = stream_graph_updates_no_mem_no_stream(user_input, graph)
            display_response(events)

def run_cli_chatbot_no_stream(model):
    graph_builder = build_graph(model)
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "1"}}
    graph = graph_builder.compile(memory)
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        events = stream_graph_updates_no_stream(user_input, graph, config)
        display_response(events)

def run_cli_chatbot_no_mem(model):
    graph_builder = build_graph(model)
    graph = graph_builder.compile()
    while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            events = stream_graph_updates_no_mem(user_input, graph)
            stream_response(events)

def run_cli_chatbot(model):
    graph_builder = build_graph(model)
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "1"}}
    graph = graph_builder.compile(memory)
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        events = stream_graph_updates(user_input, graph, config)
        stream_response(events)

def run_gradio_no_mem_no_stream(model):
    graph_builder = build_graph(model)
    graph = graph_builder.compile()
    def gradio_completion(history, system):
        if not history:
            return []
        msg = history[-1]["content"]
        events = stream_graph_updates_no_mem_no_stream(msg, graph, system=system)
        for event in events:
            for value in event.values():
                return [[gr.ChatMessage(
                            role="assistant",
                            content=value["messages"][-1].content
                        )]]
    launch_gradio_chatbot(gradio_completion)

def run_gradio_no_stream(model):
    graph_builder = build_graph(model)
    graph = graph_builder.compile()
    def gradio_completion(history, system):
        if not history:
            return []
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        if system:
            messages = [{"role": "system", "content": system}] + messages
        events = graph.stream({"messages": messages})
        for event in events:
            for value in event.values():
                return [[gr.ChatMessage(
                            role="assistant",
                            content=value["messages"][-1].content
                        )]]
    launch_gradio_chatbot(gradio_completion)

def run_gradio_no_mem(model):
    graph_builder = build_graph(model)
    graph = graph_builder.compile()
    def gradio_completion(history, system):
        if not history:
            return []
        msg = history[-1]["content"]
        events = stream_graph_updates_no_mem(msg, graph, system=system)
        output_message = ""
        for message_chunk, metadata in events:
            if message_chunk.content:
                output_message += message_chunk.content
                yield [gr.ChatMessage(
                            role="assistant",
                            content=output_message
                        )]
    launch_gradio_chatbot(gradio_completion)

def run_gradio(model):
    graph_builder = build_graph(model)
    graph = graph_builder.compile()
    def gradio_completion(history, system):
        if not history:
            return []
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        if system:
            messages = [{"role": "system", "content": system}] + messages
        events = graph.stream({"messages": messages}, stream_mode="messages")
        output_message = ""
        for message_chunk, metadata in events:
            if message_chunk.content:
                output_message += message_chunk.content
                yield [gr.ChatMessage(
                            role="assistant",
                            content=output_message
                        )]
    launch_gradio_chatbot(gradio_completion)

# Main function to run chatbot
def main(preset: str = "qwen", test: bool = False, cli: bool = False, nostream: bool = False, nomemory: bool = False):
    """
    Usage Examples:

    # Basic Test Mode (checks if the model is working)
    python chatbotnotool.py --test

    # Run CLI Mode (command-line interface)
    python chatbotnotool.py --cli

    # Run CLI Mode with a preset
    python chatbotnotool.py qwen --cli

    # Run CLI Mode without Streaming and Memory (both disabled)
    python chatbotnotool.py --cli --nostream --nomemory

    # Run in Gradio Mode (web interface)
    python chatbotnotool.py

    # Run in Gradio Mode with a preset
    python chatbotnotool.py qwen

    # Run Gradio Mode without Streaming
    python chatbotnotool.py --nostream

    # Run Gradio Mode without Memory (disables memory but keeps streaming)
    python chatbotnotool.py --nomemory

    # Run Gradio Mode without Streaming and Memory (both disabled)
    python chatbotnotool.py --nostream --nomemory
    """
    if test:
        settings = load_settings(preset)
        model = get_model(settings)
        response = model.invoke([HumanMessage(content="Hi!")])
        print(response.content)
    if cli:
        settings = load_settings(preset)
        model = get_model(settings)
        if nostream:
            if nomemory:
                run_cli_chatbot_no_mem_no_stream(model)
            else:
                run_cli_chatbot_no_stream(model)
        else:
            if nomemory:
                run_cli_chatbot_no_mem(model)
            else:
                run_cli_chatbot(model)
    else:
        settings = load_settings(preset)
        model = get_model(settings)
        if nostream:
            if nomemory:
                run_gradio_no_mem_no_stream(model)
            else:
                run_gradio_no_stream(model)
        else:
            if nomemory:
                run_gradio_no_mem(model)
            else:
                run_gradio(model)

if __name__ == "__main__":
    fire.Fire(main)
