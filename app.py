import gradio as gr
from dotenv import load_dotenv
from pro_implementation.answer import answer_question

# Load environment variables
load_dotenv(override=True)

# -------------------------
# Format Retrieved Context
# -------------------------
def format_context(context):
    """
    Render retrieved chunks as HTML content with clean highlight + source info.
    """
    if not context:
        return "<h3 style='color:red;'>⚠ No context found in DB. Check collection name or ingested documents.</h3>"

    html = "<h2 style='color:#ff7800;'>📚 Relevant Retrieved Context</h2><br>"

    for doc in context:
        html += (
            f"<div style='margin-bottom:20px;'>"
            f"<strong style='color:#ff7800;'>Source:</strong> {doc.metadata.get('source','Unknown')}<br><br>"
            f"<div style='white-space:pre-wrap; font-size:14px; line-height:1.5;'>"
            f"{doc.page_content}"
            f"</div>"
            f"</div>"
            "<hr style='border: 0; border-top: 1px dashed #ff7800; margin: 20px 0;'>"
        )
    return html

# -------------------------
# Chat Flow Handler
# -------------------------
def chat(history):
    """
    Handle conversation: user asks -> RAG -> assistant answers.
    """
    user_msg = history[-1]["content"]
    conversation_before = history[:-1]

    try:
        answer, context = answer_question(user_msg, conversation_before)
    except Exception as e:
        history.append({"role": "assistant", "content": f"⚠ Error: {str(e)}"})
        return history, "<h3 style='color:red;'>An error occurred during RAG processing.</h3>"

    history.append({"role": "assistant", "content": answer})

    return history, format_context(context)


# -------------------------
# Main UI
# -------------------------
def main():
    def add_user_message(message, history):
        """Add user's message to chatbot UI."""
        return "", history + [{"role": "user", "content": message}]

    ui_theme = gr.themes.Soft(
        primary_hue="orange",
        font=["Inter", "system-ui", "sans-serif"]
    )

    with gr.Blocks(title="Ayush Personal RAG Assistant", theme=ui_theme) as ui:

        gr.Markdown("""
        # 🤖 Ayush Personal RAG Assistant  
        Ask anything about **Ayush** — powered by your custom-built RAG pipeline.
        """)

        with gr.Row():

            # Left Column — Chat
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    type="messages",
                    height=600,
                    show_copy_button=True,
                )

                user_input = gr.Textbox(
                    placeholder="Ask something like: Who is Ayush?",
                    show_label=False,
                )

            # Right Column — Context Viewer
            with gr.Column(scale=1):
                context_box = gr.Markdown(
                    "*Retrieved context will appear here.*",
                    label="📚 Retrieved Context",
                    height=600,
                )

        # Workflow: User submits → add message → chat() → display answer + context
        user_input.submit(
            add_user_message,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot],
        ).then(
            chat,
            inputs=chatbot,
            outputs=[chatbot, context_box],
        )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
