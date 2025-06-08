import gradio as gr
import requests
import os

# BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
# Use Docker hostname if running inside Docker
docker_backend_host = "http://chatbot-backend:8000"
default_host = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Detect if running inside Docker
if os.path.exists("/.dockerenv"):
    BASE_URL = docker_backend_host
else:
    BASE_URL = default_host


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

    if "user_info" in data:
        user_info = data["user_info"]

    history.append((message, bot_reply))
    return history, ""

custom_css = """
.gr-block.gr-box {
    background-color: #f7f7f7;
    border-radius: 12px;
    padding: 20px;
    direction: rtl;
    text-align: right;
    font-family: 'Segoe UI', sans-serif;
}
.gr-chatbot {
    font-size: 16px;
    border-radius: 10px;
    direction: rtl;
    text-align: right;
}
.gr-textbox textarea {
    direction: rtl;
    text-align: right;
}
.gr-button {
    direction: rtl;
    text-align: center;
}
.big-header {
    font-size: 36px;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 10px;
    color: #333;
    text-align: center; /* שינוי: כותרת למרכז */
}
.welcome-text {
    font-size: 18px;
    color: #555;
    text-align: right;
}
#footer-logo {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}
#health-logos {
    display: flex;
    justify-content: center;
    gap: 25px;
    margin-bottom: 10px;
}
"""

with gr.Blocks(css=custom_css, title="עוזר רפואי חכם") as demo:
    with gr.Column():
        with gr.Row(elem_id="health-logos"):
            gr.Image("content/Images/macabi.jpg", show_label=False, show_download_button=False, height=60)
            gr.Image("content/Images/clalit.png", show_label=False, show_download_button=False, height=60)
            gr.Image("content/Images/meuhedet.png", show_label=False, show_download_button=False, height=60)

        gr.HTML('<div class="big-header">🤖 עוזר רפואי חכם</div>')
        gr.HTML('<div class="welcome-text">ברוכים הבאים! כאן תוכלו לשאול שאלות ולקבל מידע על שירותי בריאות מקופות החולים בישראל.</div>')

    chatbot = gr.Chatbot(height=500, label="עוזר הבריאות")

    with gr.Row():
        msg = gr.Textbox(
            show_label=False,
            placeholder="הקלידו את ההודעה כאן ולחצו Enter או על כפתור שלח...",
            scale=9
        )
        send = gr.Button("שלח", scale=1)

    send.click(fn=chat_with_bot, inputs=msg, outputs=[chatbot, msg])
    msg.submit(fn=chat_with_bot, inputs=msg, outputs=[chatbot, msg])

    with gr.Row(elem_id="footer-logo"):
        gr.Image("content/Images/kpmg-logo.png", show_label=False, show_download_button=False, height=40)

# demo.launch()

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_PORT", 8501))
    )
