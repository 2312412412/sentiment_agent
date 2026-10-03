#!/usr/bin/env python3
"""Minimal Gradio smoke test."""
import gradio as gr


def greet(name):
    return f"Hello {name}!"


with gr.Blocks() as demo:
    gr.Markdown("# Gradio Smoke Test")
    inp = gr.Textbox(label="Name")
    out = gr.Textbox(label="Greeting")
    btn = gr.Button("Run")
    btn.click(greet, inp, out)


if __name__ == "__main__":
    print("Starting local Gradio smoke test at http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
