# main.py - Fixed Gradio App
import gradio as gr
from gradio_pdf import PDF
import json
import os

from extract_fields_with_gpt import run_extraction_pipeline

def process_pdf(file):
    if not file or not file.lower().endswith(".pdf"):
        return "Please upload a valid PDF file."

    try:
        result = run_extraction_pipeline(file)

        try:
            json.loads(result)  # check if valid JSON
            return result
        except json.JSONDecodeError:
            return result

    except Exception as e:
        return f"An error occurred while processing the PDF:\n{str(e)}"


# Create Gradio interface
with gr.Blocks(title="PDF Field Extraction") as demo:
    gr.Markdown("## 📄 Upload a PDF and extract structured data")
    gr.Markdown("The prompts are built into the app. Just upload your PDF and click.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf = PDF(label="Upload a PDF")

        with gr.Column(scale=1):
            output = gr.Textbox(label="Extraction Result", lines=20, show_copy_button=True)

    btn = gr.Button("Extract Data", variant="primary")
    btn.click(fn=process_pdf, inputs=pdf, outputs=output)


if __name__ == "__main__":
    demo.launch(share=True, debug=True)