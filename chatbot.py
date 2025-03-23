import fire
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage

import gradio as gr
from gradio_chatbot_UI import ChatInterfaceCustom

from tools import multiply, add, current_time, random_number

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph import add_messages
from langgraph.prebuilt import create_react_agent

import re

from debug import chunkdebug as cc
debug_stream = False

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

from models import get_model, load_settings, load_api_key

from langchain_community.tools import BraveSearch

from debug import timer
    
def stream_response(events):
    if debug_stream:
        cc.init()

    current_conversation = []
    partial_message_content = ""


    for stream_mode, chunk in events:
        if debug_stream:
            cc(stream_mode)
            cc(chunk)
        if stream_mode == "updates":
            for key in chunk:
                if key == "agent" or key == "tools":
                    for message in chunk[key]["messages"]:
                        current_conversation.append(message)
                        partial_message_content = ""
                else:
                    raise ValueError(f"Unknown key in updates: {key}")

            partial_message_content = ""
        elif stream_mode == "messages":
            message_chunk, metadata = chunk
            if message_chunk.content:
                partial_message_content += message_chunk.content
        else:
            pass
        current_messages = current_conversation if not partial_message_content else current_conversation + [AIMessage(content=partial_message_content)]

        if debug_stream:
            cc(current_messages)

        yield current_messages

def parse_thinking_segments(text):
    """
    Given a string that may contain <think>...</think> segments,
    return a list of tuples (content, is_thinking), where:
    - content is the substring
    - is_thinking is a boolean indicating whether the substring
        is inside <think>...</think> or not.

    If <think> is unclosed, treat everything from <think> onward
    as one thinking segment.
    """
    parts = re.split(r'(<think>|</think>)', text)
    result = []
    is_thinking = False
    buffer = ''

    for part in parts:
        if part == '<think>':
            if buffer:
                result.append((buffer, is_thinking))
                buffer = ''
            is_thinking = True
        elif part == '</think>':
            if buffer:
                result.append((buffer, is_thinking))
                buffer = ''
            is_thinking = False
        else:
            buffer += part

    if buffer:
        result.append((buffer, is_thinking))

    return result

def transform_for_gradio(messages_list):
    gradio_messages = []

    for item in messages_list:
        if isinstance(item, AIMessage):
            original_content = item.content
            # Parse out <think> segments
            parsed_segments = parse_thinking_segments(original_content)
            # For each segment, create an appropriate ChatMessage
            for seg_content, is_thinking in parsed_segments:
                seg_content_stripped = seg_content  # or seg_content.strip(), if desired

                if is_thinking:
                    # Create a "Thinking..." segment
                    gradio_messages.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=seg_content_stripped,
                            metadata={"title": "Thinking..."}
                        )
                    )
                else:
                    # Normal text
                    gradio_messages.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=seg_content_stripped
                        )
                    )
            if hasattr(item, "additional_kwargs"):
                if item.additional_kwargs is not None:
                    tool_calls = item.additional_kwargs.get("tool_calls", [])
                    for tool_call in tool_calls:
                        gradio_messages.append(
                            gr.ChatMessage(
                                role="assistant",
                                content= tool_call['function']['arguments'],
                                metadata={
                                    "title": f"Using tool {tool_call['function']['name']}",
                                    "id": tool_call['id']
                                }
                            )
                        )

        elif isinstance(item, ToolMessage):
            # Just store the content as-is
            content_val = item.content
            gradio_messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=content_val,
                    metadata={
                        "title": "Tool result:",
                        "parent_id": item.tool_call_id
                    }
                )
            )

        else:
            raise ValueError(f"Unknown message type: {type(item)}")

    return gradio_messages


def gradio_history_to_langchain_history(gradio_history):
    lc_messages = []

    for entry in gradio_history:
        role = entry.get("role")
        content = entry.get("content", "")
        metadata = entry.get("metadata") or {}

        # --- If it's from the user, create a HumanMessage
        if role == "user":
            lc_messages.append(
                HumanMessage(content=content)
            )
            continue

        # Otherwise it's from the assistant
        title = metadata.get("title", "")
        if title.startswith("Using tool "):
            # This implies a "tool call"
            # Example: {"title": "Using tool multiply", "id": "101665233"}
            function_name = title[len("Using tool "):]
            tool_id = metadata.get("id", "")
            arguments = content  # Usually a JSON string

            # Create an AIMessage with a "tool_calls" entry
            lc_messages.append(
                AIMessage(
                    content="",  # tool calls often have empty "final" content
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": arguments
                                }
                            }
                        ]
                    }
                )
            )

        elif title == "Tool result:":
            # This implies the result of a tool call
            # Example: {"title": "Tool result:", "parent_id": "101665233"}
            parent_id = metadata.get("parent_id", "")
            lc_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=parent_id
                )
            )

        else:
            # Otherwise, it's a plain assistant message
            lc_messages.append(
                AIMessage(content=content)
            )

    return lc_messages

def main(preset: str = "gemma"):
    settings = load_settings(preset)
    model = get_model(settings)
    search = BraveSearch.from_api_key(api_key=load_api_key("brave"), search_kwargs={"count": 3})
    tools = [search, current_time]
    agent_executor = create_react_agent(model, tools)
    agent_graph = StateGraph(State)
    agent_graph = agent_graph.add_node("main_agent", agent_executor)
    agent_graph = agent_graph.add_edge(START, "main_agent")
    agent_graph = agent_graph.compile()

    def gradio_completion(history, system):
        history = gradio_history_to_langchain_history(history)
        if system:
            history = [SystemMessage(content=system)] + history
        events = agent_executor.stream(
            {"messages": history},
            stream_mode=["updates", "messages"],
        )
        for output in stream_response(events):
            yield transform_for_gradio(output)

    demo = ChatInterfaceCustom(gradio_completion)
    demo.launch()

if __name__ == "__main__":
    fire.Fire(main)
