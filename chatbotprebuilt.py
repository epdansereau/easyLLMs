# Less customizable script using langgraph.prebuilt and gr.ChatInterface

import fire

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

import gradio as gr
from tools import multiply, add, current_time, random_number

from langgraph.prebuilt import create_react_agent

from models import get_model, load_settings

def stream_response_cli(events):
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
    
def stream_response(events):
    """
    Streams assistant text as it arrives but instead of printing,
    accumulates all output into a list. For normal text, each
    consecutive run is appended into a single "normal_text" element.

    For example, if there's never a tool call or tool result, the final
    list would contain exactly one element, like:
    [
      {
        "type": "normal_text",
        "content": "All the text that streamed so far..."
      }
    ]

    result_list will look like:
    [
      {"type": "normal_text", "content": "some text..."},
      {"type": "tool_call", "tool_call_id": 123, "name": "my_function", "content": "JSON snippet..."},
      {"type": "tool_result", "tool_call_id": 123, "content": "...output..."},
      {"type": "normal_text", "content": "some more text..."},
      ...
    ]
    """

    # This list holds all items so far
    result_list = []

    # Track the current tool call being assembled
    current_tool_id = None
    partial_call_args = ""
    partial_call_name = None

    def finalize_tool_call():
        """
        Append the accumulated tool-call JSON for current_tool_id, if any,
        and reset the accumulation buffers.
        """
        nonlocal current_tool_id, partial_call_args, partial_call_name, result_list
        if current_tool_id is not None:
            result_list.append({
                "type": "tool_call",
                "tool_call_id": current_tool_id,
                "name": partial_call_name or "",
                "content": partial_call_args
            })
            current_tool_id = None
            partial_call_args = ""
            partial_call_name = None

    for message_chunk, metadata in events:
        # Check if this is a tool-result message
        if getattr(message_chunk, "name", None) and hasattr(message_chunk, "tool_call_id"):
            # Before storing a tool result, finalize any in-progress tool call
            finalize_tool_call()

            # Store the tool's result
            tool_call_id = message_chunk.tool_call_id
            tool_output = message_chunk.content
            result_list.append({
                "type": "tool_result",
                "tool_call_id": tool_call_id,
                "content": tool_output
            })

        else:
            # Handle any partial tool-call arguments
            calls = message_chunk.additional_kwargs.get("tool_calls", [])
            for c in calls:
                if c.get("type") == "function":
                    call_id = c.get("id")
                    function_data = c.get("function", {})
                    function_name = function_data.get("name", "")
                    args_piece = function_data.get("arguments", "")

                    # If we see a new call ID, finalize the old one
                    # and start a new accumulation
                    if call_id is not None and call_id != current_tool_id:
                        finalize_tool_call()
                        current_tool_id = call_id
                        partial_call_args = args_piece
                        partial_call_name = function_name
                    else:
                        # Same call ID -> keep appending to the last known call
                        partial_call_args += args_piece
                        # If a function name was provided in this snippet, store/update it.
                        # Typically, it's the same name across partial chunks, but we'll store anyway:
                        if function_name:
                            partial_call_name = function_name

            # If there's normal text in this chunk, we add/append it
            if message_chunk.content:
                # Check if the last item in the list is also normal_text
                if result_list and result_list[-1]["type"] == "normal_text":
                    # Append to the previous normal_text
                    result_list[-1]["content"] += message_chunk.content
                else:
                    # Create a new normal_text entry
                    result_list.append({
                        "type": "normal_text",
                        "content": message_chunk.content
                    })

        # After processing this chunk, yield the entire list so far
        yield result_list

    # After all chunks, finalize any leftover tool-call
    finalize_tool_call()

    # Yield the final state of the list once more
    yield result_list


def transform_for_gradio(messages_list):

    gradio_messages = []

    for item in messages_list:
        msg_type = item.get("type")

        # Normal text => just put all text into the content
        if msg_type == "normal_text":
            gradio_messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=item.get("content", "")
                )
            )

        elif msg_type == "tool_call":
            # Use the tool name in the metadata's title
            gradio_messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=item.get("content", ""),  # e.g. JSON snippet
                    metadata={
                        "title": f"Using tool {item.get('name', '')}",
                        "id": item.get("tool_call_id", "")
                    }
                )
            )

        elif msg_type == "tool_result":
            # Parse content as integer if possible
            content_val = item.get("content", "")
            gradio_messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=content_val,
                    metadata={
                        "title": "Tool result:",
                        "parent_id": item.get("tool_call_id", "")
                    }
                )
            )

        else:
            # If there's some other type you haven't handled explicitly,
            # decide what to do (ignore, or treat as normal text, etc.)
            # For safety, let's just treat it as normal text:
            gradio_messages.append(
                gr.ChatMessage(role="assistant", content=item.get("content", ""))
            )

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

def run_cli_chatbot(model):
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "1"}}
    tools = [multiply, add, current_time, random_number]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

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
        stream_response_cli(events)

def run_gradio(model):
    tools = [multiply, add, current_time, random_number]
    agent_executor = create_react_agent(model, tools)
    def gradio_completion(message, history):
        events = agent_executor.stream(
            {"messages": gradio_history_to_langchain_history(history) + [HumanMessage(content=message)]},
            stream_mode="messages",
        )
        for output in stream_response(events):
            yield transform_for_gradio(output)
    gr.ChatInterface(
        fn=gradio_completion, 
        type="messages",
    ).launch()

def main(preset: str = "qwen", cli: bool = False):
    settings = load_settings(preset)
    model = get_model(settings)
    if cli:
        run_cli_chatbot(model)
    else:
        run_gradio(model)

if __name__ == "__main__":
    fire.Fire(main)
