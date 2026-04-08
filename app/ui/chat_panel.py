"""Chat panel component for Lullus.

Provides a conversational interface backed by the RAG engine, with
source citations, quick-action buttons, and conversation export.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# Quick-action button definitions
QUICK_ACTIONS: list[dict[str, str]] = [
    {"label": "Explain topic", "icon": "\U0001F4A1", "prefix": "Explain the following topic in detail: "},
    {"label": "Generate exercise", "icon": "\u270F\uFE0F", "prefix": "Generate a practice exercise about: "},
    {"label": "Check understanding", "icon": "\U0001F9EA", "prefix": "Quiz me to check my understanding of: "},
    {"label": "Find resources", "icon": "\U0001F50D", "prefix": "Find and summarise relevant resources about: "},
]

# Retrieval mode definitions
RETRIEVAL_MODES = {
    "\U0001F3AF Precise": "precise",
    "\U0001F4D6 Exhaustive": "exhaustive",
}


def _init_chat_state() -> None:
    """Initialise chat-related session state keys."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_input_prefix" not in st.session_state:
        st.session_state.chat_input_prefix = ""
    if "chat_processing" not in st.session_state:
        st.session_state.chat_processing = False


def _get_rag_engine() -> Any:
    """Retrieve the RAG engine from session state."""
    return st.session_state.get("rag_engine")


def _process_message(question: str, retrieval_mode: str = "precise") -> None:
    """Process a user message: call RAG and store the response."""
    st.session_state.chat_processing = True

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat(),
            "id": str(uuid.uuid4()),
        }
    )

    rag = _get_rag_engine()
    sources: list[dict[str, str]] = []
    confidence: float = 0.0
    answer_text: str = ""

    if rag is None:
        answer_text = (
            "The RAG engine is not available. Please ensure Ollama is running "
            "and check the Settings page."
        )
    else:
        try:
            response = rag.query(
                question, mode="chat", retrieval_mode=retrieval_mode
            )
            answer_text = response.answer
            sources = [
                {
                    "title": getattr(s, "title", getattr(s, "filename", str(s))),
                    "content": str(s),
                }
                for s in (response.sources or [])
            ]
            confidence = response.confidence_score
        except Exception as exc:
            logger.error("RAG query failed: %s", exc)
            answer_text = f"Sorry, an error occurred: {exc}"

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "id": str(uuid.uuid4()),
            "retrieval_mode": retrieval_mode,
        }
    )
    st.session_state.chat_processing = False


def _render_message(msg: dict[str, Any]) -> None:
    """Render a single chat message with avatar and optional sources."""
    avatar = "\U0001F9D1\u200D\U0001F393" if msg["role"] == "user" else "\U0001F4DA"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            sources = msg.get("sources", [])
            confidence = msg.get("confidence", 0.0)

            col_copy, col_conf = st.columns([1, 1])
            with col_conf:
                if confidence > 0:
                    conf_pct = int(confidence * 100)
                    conf_color = "#4ade80" if conf_pct >= 70 else "#facc15" if conf_pct >= 40 else "#f87171"
                    st.markdown(
                        f'<span style="color:{conf_color};font-size:0.8rem;">'
                        f"Confidence: {conf_pct}%</span>",
                        unsafe_allow_html=True,
                    )

            if sources:
                with st.expander(f"\U0001F4C4 Sources ({len(sources)})"):
                    for idx, src in enumerate(sources, 1):
                        title = src.get("title", f"Source {idx}")
                        st.markdown(f"**{idx}. {title}**")
                        content_preview = src.get("content", "")[:300]
                        if content_preview:
                            st.caption(content_preview)

            with col_copy:
                if st.button("\U0001F4CB Copy", key=f"copy_{msg.get('id', '')}"):
                    st.code(msg["content"], language=None)
                    st.toast("Response displayed in copyable block.")


def _export_conversation() -> str:
    """Export the conversation history as a Markdown string."""
    lines: list[str] = ["# Lullus Conversation Export", ""]
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in st.session_state.get("chat_messages", []):
        role = "User" if msg["role"] == "user" else "Lullus"
        timestamp = msg.get("timestamp", "")
        lines.append(f"### {role}")
        if timestamp:
            lines.append(f"*{timestamp}*")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")

        if msg["role"] == "assistant" and msg.get("sources"):
            lines.append("**Sources:**")
            for src in msg["sources"]:
                lines.append(f"- {src.get('title', 'Unknown source')}")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def render_chat_panel() -> None:
    """Render the complete chat panel UI."""
    _init_chat_state()

    st.markdown(
        '<h1 class="page-title">\U0001F4AC Chat with Lullus</h1>',
        unsafe_allow_html=True,
    )

    # Check knowledge base status
    embedding_manager = st.session_state.get("embedding_manager")
    has_documents = False
    if embedding_manager is not None:
        try:
            docs = embedding_manager.get_all_documents()
            has_documents = len(docs) > 0
        except Exception:
            pass

    if not has_documents:
        st.info(
            "Your knowledge base is empty. Upload documents in the **Knowledge Base** "
            "tab to get context-aware answers. You can still chat, but responses "
            "will not reference course materials.",
            icon="\U0001F4DA",
        )

    # Retrieval mode selector
    col_mode, col_spacer = st.columns([2, 3])
    with col_mode:
        mode_label = st.radio(
            "Answer mode",
            list(RETRIEVAL_MODES.keys()),
            horizontal=True,
            key="retrieval_mode_radio",
            help=(
                "**Precise** — Short, direct, factual answers. "
                "**Exhaustive** — Long, comprehensive, detailed answers. "
                "Both answer only from your Knowledge Base."
            ),
        )
    retrieval_mode = RETRIEVAL_MODES[mode_label]

    # Quick action buttons
    st.markdown('<div class="quick-actions-row">', unsafe_allow_html=True)
    cols = st.columns(len(QUICK_ACTIONS))
    for i, action in enumerate(QUICK_ACTIONS):
        with cols[i]:
            if st.button(
                f"{action['icon']} {action['label']}",
                key=f"qa_{action['label']}",
                use_container_width=True,
            ):
                st.session_state.chat_input_prefix = action["prefix"]
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Export button row
    col_export, col_clear = st.columns([1, 1])
    with col_export:
        if st.session_state.chat_messages:
            export_md = _export_conversation()
            st.download_button(
                label="\U0001F4E5 Export conversation",
                data=export_md,
                file_name=f"lullus_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
    with col_clear:
        if st.session_state.chat_messages:
            if st.button("\U0001F5D1\uFE0F Clear conversation", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()

    # Message history
    message_container = st.container()
    with message_container:
        if not st.session_state.chat_messages:
            st.markdown(
                '<div class="empty-chat">'
                "<h3>Welcome to Lullus!</h3>"
                "<p>Ask me anything about your uploaded materials. "
                "Use the quick action buttons above or type your question below.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_messages:
                _render_message(msg)

    # Chat input
    prefix = st.session_state.get("chat_input_prefix", "")
    if prefix:
        st.session_state.chat_input_prefix = ""

    user_input = st.chat_input(
        placeholder="Ask a question about your materials...",
        key="chat_input_widget",
    )

    if user_input:
        full_message = prefix + user_input if prefix else user_input
        with st.spinner("Thinking..."):
            _process_message(full_message, retrieval_mode=retrieval_mode)
        st.rerun()
