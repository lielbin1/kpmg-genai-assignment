import gradio as gr
import requests
import os

BASE_URL = os.getenv("API_BASE_URL")

history = []
user_info = {}
phase = "collect_info"
current_language = None

def chat_with_bot(message):
    global history, user_info, phase, current_language

    payload = {
        "user_message": message,
        "history": history,
        "user_info": user_info,
        "language": current_language or ""
    }

    response = requests.post(f"{BASE_URL}/ask", json=payload)
    data = response.json()
    bot_reply = data.get("response", "שגיאה")

    if current_language is None and "language" in data:
        current_language = data["language"]

    history.append((message, bot_reply))
    return history, ""

with gr.Blocks(css=".gr-chatbot {font-size: 16px;}") as demo:
    gr.Markdown("## 🩺 Chatbot for Israeli Healthcare Services")

    chatbot = gr.Chatbot(height=550, label="Healthcare Assistant")

    with gr.Row():
        msg = gr.Textbox(
            show_label=False,
            placeholder="Type your message here and press Enter or click Send...",
            scale=9
        )
        send = gr.Button("Send", scale=1)

    send.click(fn=chat_with_bot, inputs=msg, outputs=[chatbot, msg])
    msg.submit(fn=chat_with_bot, inputs=msg, outputs=[chatbot, msg])

demo.launch()
