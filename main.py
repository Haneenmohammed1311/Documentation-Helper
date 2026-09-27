from typing import Any, Dict, List

import streamlit as st

from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    """
    Take the raw list of retrieved Document objects returned by run_llm
    and pull out just the source URL from each one's metadata, so the
    UI can display a clean list of links instead of full document text.
    """
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


def _render_message(msg: Dict[str, Any]) -> None:
    """
    Display a single chat message in the conversation, including its
    sources in a collapsible expander if any were attached to it.
    """
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")


def _clear_chat() -> None:
    """
    Reset the conversation back to its initial empty state and
    immediately rerun the app so the cleared UI shows right away.
    """
    st.session_state.pop("messages", None)
    st.rerun()


def _init_chat_state() -> None:
    """
    Make sure a messages list exists in session state before the app
    tries to read from it, seeding it with a single welcome message
    on the very first load.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me anything about LangChain docs. I'll retrieve relevant context and cite sources.",
                "sources": [],
            }
        ]


def _handle_user_prompt(prompt: str) -> None:
    """
    Add the user's question to the conversation, call the RAG pipeline
    through run_llm, and append the assistant's answer, along with its
    sources, back into the conversation once it's ready.
    """
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer..."):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)


# Page setup
st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")
st.caption("Running on Chroma, local HuggingFace embeddings, and Gemini")

# Sidebar, session controls and a couple of example questions to try
with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        _clear_chat()

    st.subheader("Try asking")
    example_questions = [
        "What is LangChain?",
        "What are agents in LangChain?",
        "How does structured output work?",
    ]
    example_clicked = None
    for question in example_questions:
        if st.button(question, use_container_width=True):
            example_clicked = question

_init_chat_state()

# Render the full conversation so far
for msg in st.session_state.messages:
    _render_message(msg)

# A clicked example question is treated exactly like typed input
prompt = st.chat_input("Ask a question about LangChain...") or example_clicked
if prompt:
    _handle_user_prompt(prompt)