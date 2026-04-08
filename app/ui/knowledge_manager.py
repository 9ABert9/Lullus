"""
Knowledge base manager UI for Lullus.

Provides drag-and-drop file upload with discipline-specific chunking
strategy selection, document listing, and indexing controls.
"""

import logging
from pathlib import Path

import streamlit as st

from app.core.embedding_manager import CHUNKING_STRATEGIES

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = [".pdf", ".docx", ".pptx", ".txt", ".md", ".epub", ".html", ".csv"]
FORMAT_ICONS = {
    ".pdf": "\U0001F4D5",
    ".docx": "\U0001F4D8",
    ".pptx": "\U0001F4D9",
    ".txt": "\U0001F4C4",
    ".md": "\U0001F4DD",
    ".epub": "\U0001F4DA",
    ".html": "\U0001F310",
    ".csv": "\U0001F4CA",
}


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def render_knowledge_manager(embedding_manager, file_watcher=None) -> None:
    """Render the knowledge base management page.

    Args:
        embedding_manager: EmbeddingManager instance for indexing operations.
        file_watcher: Optional FileWatcher instance for status display.
    """
    st.markdown("## \U0001F4DA Knowledge Base")
    st.markdown("Upload your course materials \u2014 PDFs, slides, notes, and more. "
                "They'll be automatically indexed for the AI to reference.")

    # --- File watcher status ---
    if file_watcher:
        status = file_watcher.get_status()
        col1, col2, col3 = st.columns(3)
        with col1:
            running = status.get("is_running", False)
            watcher_label = "\U0001F7E2 Active" if running else "\U0001F534 Stopped"
            st.markdown(f"**Watcher:** {watcher_label}")
        with col2:
            st.markdown(f"**Files processed:** {status.get('files_processed', 0)}")
        with col3:
            last = status.get("last_event")
            st.markdown(f"**Last event:** {last or 'None'}")

        watch_path = status.get("watch_path", "./knowledge_base")
        st.info(
            f"\U0001F4C1 **Watched folder:** `{watch_path}`\n\n"
            f"You can also drop files directly into this folder \u2014 they'll be auto-indexed."
        )

    st.markdown("---")

    # --- Chunking strategy selector ---
    st.markdown("### Upload Files")

    strategy_labels = {k: v["label"] for k, v in CHUNKING_STRATEGIES.items()}
    strategy_descriptions = {k: v["description"] for k, v in CHUNKING_STRATEGIES.items()}

    selected_strategy_label = st.radio(
        "Chunking strategy",
        list(strategy_labels.values()),
        horizontal=True,
        key="chunking_strategy_radio",
        help="Choose how uploaded documents are split into chunks for retrieval.",
    )

    # Reverse-lookup the key from the label
    selected_strategy = "auto"
    for key, label in strategy_labels.items():
        if label == selected_strategy_label:
            selected_strategy = key
            break

    st.caption(strategy_descriptions.get(selected_strategy, ""))

    # --- File upload ---
    uploaded_files = st.file_uploader(
        "Drag and drop your course materials here",
        type=[fmt.lstrip(".") for fmt in SUPPORTED_FORMATS],
        accept_multiple_files=True,
        key="knowledge_uploader",
    )

    if uploaded_files:
        kb_path = Path("./knowledge_base")
        kb_path.mkdir(parents=True, exist_ok=True)

        progress = st.progress(0, text="Processing uploads...")
        total = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files):
            file_path = kb_path / uploaded_file.name
            progress.progress(
                (i) / total,
                text=f"Processing {uploaded_file.name}..."
            )

            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                embedding_manager.add_document(str(file_path), strategy=selected_strategy)
                st.success(f"\u2713 Indexed: {uploaded_file.name} (strategy: {selected_strategy})")
            except Exception as e:
                st.error(f"\u2717 Failed to process {uploaded_file.name}: {e}")
                logger.error("Upload processing error for %s: %s", uploaded_file.name, e)

        progress.progress(1.0, text="All files processed!")

    st.markdown("---")

    # --- Indexed documents list ---
    st.markdown("### Indexed Documents")

    col_refresh, col_reindex = st.columns([1, 1])
    with col_refresh:
        if st.button("\U0001F504 Refresh List", use_container_width=True):
            st.rerun()
    with col_reindex:
        if st.button("\U0001F4E6 Re-index All", use_container_width=True):
            with st.spinner("Re-indexing all documents..."):
                try:
                    embedding_manager.reindex_all()
                    st.success("All documents re-indexed successfully!")
                except Exception as e:
                    st.error(f"Re-indexing failed: {e}")
            st.rerun()

    # Get all indexed documents
    try:
        documents = embedding_manager.get_all_documents()
    except Exception as e:
        st.error(f"Could not load documents: {e}")
        documents = []

    if not documents:
        st.markdown(
            """
            <div style="text-align: center; padding: 40px 20px;
                        border: 2px dashed #555; border-radius: 12px;
                        margin: 20px 0;">
                <p style="font-size: 1.3em; margin-bottom: 8px;">\U0001F4C2 No documents indexed yet</p>
                <p style="color: #888;">Upload files above or drop them into the
                <code>knowledge_base/</code> folder to get started.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(f"**{len(documents)} document(s) indexed**")

    # Display as card grid (2 columns)
    for row_start in range(0, len(documents), 2):
        cols = st.columns(2)
        for col_idx in range(2):
            doc_idx = row_start + col_idx
            if doc_idx >= len(documents):
                break

            doc = documents[doc_idx]
            with cols[col_idx]:
                ext = Path(doc.filename).suffix.lower() if hasattr(doc, "filename") else ""
                icon = FORMAT_ICONS.get(ext, "\U0001F4C4")
                file_size = _format_file_size(doc.file_size) if hasattr(doc, "file_size") else "N/A"
                chunks = getattr(doc, "chunk_count", 0)
                date_added = getattr(doc, "date_indexed", "Unknown")
                file_type = getattr(doc, "file_type", ext.upper())

                st.markdown(
                    f"""
                    <div style="background: #1e1e2e; border: 1px solid #333;
                                border-radius: 10px; padding: 16px; margin-bottom: 12px;">
                        <div style="font-size: 1.1em; font-weight: 600; margin-bottom: 8px;">
                            {icon} {doc.filename}
                        </div>
                        <div style="color: #aaa; font-size: 0.85em;">
                            \U0001F4E6 {chunks} chunks &nbsp;|&nbsp;
                            \U0001F4BE {file_size} &nbsp;|&nbsp;
                            \U0001F3F7\uFE0F {file_type}
                        </div>
                        <div style="color: #888; font-size: 0.8em; margin-top: 4px;">
                            Added: {date_added}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"\U0001F5D1\uFE0F Remove",
                    key=f"delete_{doc.doc_id}",
                    use_container_width=True,
                ):
                    try:
                        embedding_manager.remove_document(doc.doc_id)
                        file_path = Path("./knowledge_base") / doc.filename
                        if file_path.exists():
                            file_path.unlink()
                        st.success(f"Removed {doc.filename}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to remove: {e}")

    # --- File watcher log ---
    if file_watcher:
        with st.expander("\U0001F4CB Watcher Activity Log"):
            log = file_watcher.get_log()
            if not log:
                st.markdown("*No events recorded yet.*")
            else:
                for event in log[:20]:
                    status_icon = "\u2705" if event["status"] == "success" else "\u274C"
                    st.markdown(
                        f"`{event['timestamp']}` {status_icon} "
                        f"**{event['event_type']}** \u2014 {event['message']}"
                    )
