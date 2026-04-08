"""
Smart Notes UI panel for Lullus.

Lets users paste rough notes, keywords, or broken sentences.
The assistant produces polished text enriched with Knowledge Base content.
"""

import logging

import streamlit as st

logger = logging.getLogger(__name__)

STYLE_OPTIONS = {
    "Detailed": "detailed",
    "Concise": "concise",
    "Outline": "outline",
}


def render_smart_notes(smart_notes_engine) -> None:
    """Render the Smart Notes page.

    Args:
        smart_notes_engine: SmartNotes instance.
    """
    st.markdown("## \u2728 Smart Notes")
    st.info(
        "Paste your rough notes, keywords, or broken sentences below. "
        "The assistant will **transform them into polished, readable text** "
        "and **enrich them with relevant information from your Knowledge Base**, "
        "filling in gaps and adding references.",
        icon="\U0001F4DD",
    )

    # Check KB status
    embedding_manager = st.session_state.get("embedding_manager")
    if embedding_manager:
        try:
            doc_count = len(embedding_manager.get_all_documents())
        except Exception:
            doc_count = 0
        if doc_count == 0:
            st.warning(
                "Your Knowledge Base is empty. Upload course materials first "
                "so Smart Notes can enrich your text with relevant content."
            )
        else:
            st.caption(f"\U0001F4DA {doc_count} document(s) available for enrichment")

    st.markdown("---")

    # Input area
    raw_notes = st.text_area(
        "Your rough notes",
        height=250,
        placeholder=(
            "Paste your notes here...\n\n"
            "Examples:\n"
            "- keywords, bullet points, fragments\n"
            "- half-finished sentences\n"
            "- lecture shorthand\n"
            "- mixed languages or abbreviations\n\n"
            "The assistant will turn this into clean, structured text."
        ),
        key="smart_notes_input",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        style_label = st.radio(
            "Output style",
            list(STYLE_OPTIONS.keys()),
            horizontal=True,
            key="smart_notes_style",
        )
    with col2:
        st.markdown("")  # spacing

    style = STYLE_OPTIONS[style_label]

    # Generate button
    col_gen, col_clear = st.columns([3, 1])
    with col_gen:
        generate_clicked = st.button(
            "\u2728 Enhance Notes",
            use_container_width=True,
            type="primary",
            disabled=not raw_notes or not raw_notes.strip(),
        )
    with col_clear:
        if st.button("\U0001F5D1\uFE0F Clear", use_container_width=True, key="clear_notes"):
            if "smart_notes_result" in st.session_state:
                del st.session_state.smart_notes_result
            st.rerun()

    # Process
    if generate_clicked and raw_notes and raw_notes.strip():
        with st.spinner("Enhancing your notes with Knowledge Base content..."):
            try:
                result = smart_notes_engine.enhance_notes(
                    raw_notes=raw_notes.strip(),
                    style=style,
                )
                st.session_state.smart_notes_result = result
            except Exception as e:
                st.error(f"Failed to enhance notes: {e}")
                logger.error("Smart notes enhancement failed: %s", e)

    # Display result
    result = st.session_state.get("smart_notes_result")
    if result:
        st.markdown("---")
        st.markdown("### \U0001F4C4 Enhanced Notes")

        # Metadata bar
        st.caption(f"Word count: {result.word_count}")

        # The enhanced text
        st.markdown(result.enhanced_text)

        # Sources
        if result.sources_used:
            with st.expander(f"\U0001F4DA Sources referenced ({len(result.sources_used)})"):
                for src in result.sources_used:
                    filename = src.get("filename", "Unknown")
                    page = src.get("page", "")
                    section = src.get("section", "")
                    relevance = src.get("relevance", "")
                    parts = [f"**{filename}**"]
                    if page:
                        parts.append(f"p. {page}")
                    if section:
                        parts.append(f"\u00A7 {section}")
                    if relevance:
                        parts.append(f"(relevance: {relevance})")
                    st.markdown("- " + " \u2014 ".join(parts))

        # Export options
        st.markdown("---")
        col_md, col_latex = st.columns(2)
        with col_md:
            st.download_button(
                "\U0001F4E5 Download as Markdown",
                data=result.enhanced_text,
                file_name="smart_notes.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_latex:
            from app.utils.export_utils import export_to_latex
            latex = export_to_latex(result.enhanced_text, "Smart Notes")
            st.download_button(
                "\U0001F4E5 Download as LaTeX",
                data=latex,
                file_name="smart_notes.tex",
                mime="text/plain",
                use_container_width=True,
            )
