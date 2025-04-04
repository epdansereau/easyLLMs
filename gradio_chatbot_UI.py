
import gradio as gr

class ChatInterfaceCustom(gr.Blocks):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.is_running = False
        with self:
            chatbot = gr.Chatbot(type="messages",editable='user', height="75vh", show_copy_button=True)
            msg = gr.Textbox(show_label=False, placeholder="Type a message...", container=False, submit_btn=True, stop_btn=False)
            default_system = '''You are an helpful AI agent. Use the tools at your disposal to assist the user in their queries as needed. Some replies don't require any tools, only conversation. Some replies require more than one tool. Some require you to use a tool and wait for the result before continuing your answer. Current time is {current_time}.'''
            with gr.Accordion("System prompt", open=False):
                system = gr.Textbox(value=default_system, show_label=False, placeholder="Enter a system prompt or leave empty for no system prompt...", container=False)

            def user(user_message, history: list):
                return "", history + [{"role": "user", "content": user_message}]

            def bot(history, system):
                for chunk in fn(history, system):
                    if self.is_running:
                        yield history + chunk
                    else:
                        break

            def set_running_state(msg):
                # When a run starts, hide the submit button and show the stop button.
                self.is_running = True
                return gr.Textbox(submit_btn=False, stop_btn=True)

            def set_idle_state(msg):
                # When a run ends, show the submit button and hide the stop button.
                self.is_running = False
                return gr.Textbox(submit_btn=True, stop_btn=False)

            submit_event = msg.submit(
                user, [msg, chatbot], [msg, chatbot],
            ).then(
                set_running_state, msg, msg,
            ).then(
                bot, [chatbot, system], chatbot
            ).then(
                set_idle_state, msg, msg,
            )
            msg.stop(
                set_idle_state, msg, msg,
            )