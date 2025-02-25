
import gradio as gr

def launch_gradio_chatbot(fn):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type="messages",editable=True, height="75vh")
        msg = gr.Textbox(show_label=False, placeholder="Type a message...", container=False, submit_btn=True, stop_btn=True)
        default_system = '''You are an helpful AI agent. Use the tools at your disposal to assist the user in their queries as needed. Some replies don't require any tools, only conversation. Some replies require more than one tool. Some require you to use a tool and wait for the result before continuing your answer.'''
        with gr.Accordion("System prompt", open=False):
            system = gr.Textbox(value=default_system, show_label=False, placeholder="Enter a system prompt or leave empty for no system prompt...", container=False)

        def user(user_message, history: list):
            return "", history + [{"role": "user", "content": user_message}]

        def bot(history, system):
            for chunk in fn(history, system):
                yield history + chunk

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, system], chatbot
        )
    demo.launch()