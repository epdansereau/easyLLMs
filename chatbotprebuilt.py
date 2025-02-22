import yaml
import fire
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

import gradio as gr

from tools import multiply, add, current_time, random_number

from langgraph.prebuilt import create_react_agent

# Load settings from YAML file
def load_settings(preset):
    with open("chatbotsettings.yaml", "r") as file:
        presets = yaml.safe_load(file)
    if preset not in presets:
        raise ValueError(f"Preset '{preset}' not found in settings.yaml. Available presets: {list(presets.keys())}")
    return presets[preset]

# Load API keys from apikeys.yaml
def load_api_key(provider):
    with open("apikeys.yaml", "r") as file:
        api_keys = yaml.safe_load(file)
    if provider not in api_keys:
        raise ValueError(f"API key for provider '{provider}' not found in apikeys.yaml.")
    return api_keys[provider]

# Get the model instance
def get_model(settings):
    api_key = load_api_key(settings["custom_llm_provider"])
    return ChatLiteLLM(
        model=settings["model"], 
        custom_llm_provider=settings["custom_llm_provider"], 
        api_base=settings.get("api_base"), 
        api_key=api_key
    )

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

def stream_graph_updates_managed_history(user_input: str, graph, config):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="messages"
    )
    return events

def stream_graph_updates(user_input: str, graph):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="messages"
    )
    return events

def stream_response(events):
    """
    Streams assistant text as it arrives and prints tool calls/results in a cleaned-up format.

    - Streams normal assistant text chunk-by-chunk (flushes immediately).
    - Accumulates partial tool-call JSON for each call_id. As soon as we detect the call is done,
      we print the final "[TOOL_CALL (id=...): (...)]" line.
    - Prints "[TOOL_RESULT (id=...): ...]" immediately upon receiving tool result chunks.
    """

    # Track the current tool call being assembled
    current_tool_id = None
    partial_call_args = ""

    def finalize_tool_call():
        """
        Prints the accumulated tool-call JSON for current_tool_id, if any,
        and resets the accumulation buffers.
        """
        nonlocal current_tool_id, partial_call_args
        if current_tool_id is not None:
            # Print the fully accumulated tool call line
            print(f"[TOOL_CALL (id={current_tool_id}): ({partial_call_args})]", flush=True)
            # Reset
            current_tool_id = None
            partial_call_args = ""

    for message_chunk, metadata in events:
        # Check if this is a tool-result message (often identified by `name` and `tool_call_id`)
        if getattr(message_chunk, "name", None) and hasattr(message_chunk, "tool_call_id"):
            # Before printing tool result, finalize any in-progress tool call
            finalize_tool_call()

            # Print the tool's result immediately
            tool_call_id = message_chunk.tool_call_id
            tool_output = message_chunk.content
            print(f"[TOOL_RESULT (id={tool_call_id}): {tool_output}]", flush=True)

        else:
            # Handle any partial tool-call arguments
            calls = message_chunk.additional_kwargs.get("tool_calls", [])
            for c in calls:
                if c.get("type") == "function":
                    call_id = c.get("id")
                    # The chunk might have only a piece of JSON, e.g., '{"a' or ': 999'
                    args_piece = c.get("function", {}).get("arguments", "")

                    # If we see a new, non-None call ID, finalize the old one
                    # and start accumulating the new call's arguments
                    if call_id is not None and call_id != current_tool_id:
                        finalize_tool_call()
                        current_tool_id = call_id
                        partial_call_args = args_piece
                    else:
                        # Same call ID or None -> keep appending to the last known call
                        partial_call_args += args_piece

            # If there's normal text in this chunk, print it immediately (streamed)
            if message_chunk.content:
                print(message_chunk.content, end="", flush=True)

    # After all chunks, finalize any leftover tool-call
    finalize_tool_call()

    # Optionally print a final newline
    print()

def run_cli_chatbot(model):
    tools = [multiply, add, current_time, random_number]
    memory = MemorySaver()
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        events = agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="messages",
        )
        stream_response(events)

def main(preset: str = "llama3", cli: bool = False):
    settings = load_settings(preset)
    model = get_model(settings)
    if cli:
        pass
    run_cli_chatbot(model)

if __name__ == "__main__":
    fire.Fire(main)
