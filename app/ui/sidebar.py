"""Sidebar navigation component for Lullus.

Renders the left sidebar with branding, navigation, connection status,
profile switching, and theme controls.
"""

import logging
from datetime import datetime
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Navigation pages with labels and icons
NAV_ITEMS: list[dict[str, str]] = [
    {"key": "chat", "label": "Chat", "icon": "💬"},
    {"key": "knowledge", "label": "Knowledge Base", "icon": "📚"},
    {"key": "exercises", "label": "Exercises", "icon": "✏️"},
    {"key": "knowledge_check", "label": "Knowledge Check", "icon": "🧪"},
    {"key": "homework", "label": "Homework Writer", "icon": "📝"},
    {"key": "profile", "label": "Profile", "icon": "👤"},
    {"key": "settings", "label": "Settings", "icon": "⚙️"},
]


def _init_sidebar_state() -> None:
    """Ensure all sidebar-related session state keys exist."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "last_sync_time" not in st.session_state:
        st.session_state.last_sync_time = None


def _check_ollama_status() -> tuple[bool, str]:
    """Check Ollama connection status and current model name.

    Returns:
        A tuple of (is_connected, model_name).
    """
    try:
        llm_engine = st.session_state.get("llm_engine")
        if llm_engine is None:
            return False, "Not initialised"
        connected: bool = llm_engine.check_connection()
        if connected:
            model_name: str = getattr(llm_engine, "model", "Unknown model")
            return True, model_name
        return False, "Disconnected"
    except Exception as exc:
        logger.warning("Ollama status check failed: %s", exc)
        return False, "Error"


def _get_indexed_doc_count() -> int:
    """Return the number of documents currently indexed."""
    try:
        embedding_manager = st.session_state.get("embedding_manager")
        if embedding_manager is None:
            return 0
        docs = embedding_manager.get_all_documents()
        return len(docs)
    except Exception as exc:
        logger.warning("Failed to get document count: %s", exc)
        return 0


def _get_active_profile_name() -> str:
    """Return the display name of the active profile."""
    try:
        profile_manager = st.session_state.get("profile_manager")
        if profile_manager is None:
            return "No profile"
        profile = profile_manager.get_active_profile()
        if profile is None:
            return "No profile"
        student = profile.get("student", {})
        course = profile.get("course", {})
        name = student.get("name", "Unknown")
        course_name = course.get("name", "")
        if course_name:
            return f"{name} - {course_name}"
        return name
    except Exception as exc:
        logger.warning("Failed to get active profile name: %s", exc)
        return "Error loading profile"


def _build_profile_options() -> tuple[list[str], list[str], int]:
    """Build profile options for the switcher dropdown.

    Returns:
        A tuple of (display_labels, profile_ids, active_index).
    """
    labels: list[str] = []
    ids: list[str] = []
    active_index: int = 0
    try:
        profile_manager = st.session_state.get("profile_manager")
        if profile_manager is None:
            return ["No profiles available"], [""], 0
        profiles = profile_manager.list_profiles()
        if not profiles:
            return ["No profiles available"], [""], 0
        for i, p in enumerate(profiles):
            label = f"{p['course_code']} - {p['student_name']}"
            labels.append(label)
            ids.append(p["profile_id"])
            if p.get("is_active", False):
                active_index = i
    except Exception as exc:
        logger.warning("Failed to build profile options: %s", exc)
        return ["Error loading profiles"], [""], 0
    return labels, ids, active_index


def render_sidebar() -> str:
    """Render the full sidebar and return the selected page key.

    Returns:
        The key string of the currently selected navigation page.
    """
    _init_sidebar_state()

    with st.sidebar:
        # -- Branding --
        st.markdown(
            '<div class="sidebar-brand">'
            '<span class="brand-icon">🎓</span>'
            '<span class="brand-text">Lullus</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # -- Ollama connection status --
        is_connected, model_name = _check_ollama_status()
        status_color = "#4ade80" if is_connected else "#f87171"
        status_label = "Connected" if is_connected else "Disconnected"
        st.markdown(
            f'<div class="status-row">'
            f'<span class="status-dot" style="background:{status_color};"></span>'
            f'<span class="status-text">Ollama: {status_label}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if is_connected:
            st.markdown(
                f'<div class="status-model">Model: <code>{model_name}</code></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # -- Navigation --
        st.markdown(
            '<p class="sidebar-section-label">NAVIGATION</p>',
            unsafe_allow_html=True,
        )
        for item in NAV_ITEMS:
            is_active = st.session_state.current_page == item["key"]
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                f"{item['icon']}  {item['label']}",
                key=f"nav_{item['key']}",
                use_container_width=True,
                type=btn_type,
            ):
                st.session_state.current_page = item["key"]
                logger.info("Navigated to page: %s", item["key"])
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # -- Knowledge base stats --
        doc_count = _get_indexed_doc_count()
        st.markdown(
            f'<div class="sidebar-stat">'
            f'<span class="stat-label">Indexed documents</span>'
            f'<span class="stat-value">{doc_count}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        last_sync: Optional[datetime] = st.session_state.get("last_sync_time")
        sync_display = last_sync.strftime("%H:%M:%S") if last_sync else "Never"
        st.markdown(
            f'<div class="sidebar-stat">'
            f'<span class="stat-label">Last sync</span>'
            f'<span class="stat-value">{sync_display}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # -- Profile section --
        st.markdown(
            '<p class="sidebar-section-label">PROFILE</p>',
            unsafe_allow_html=True,
        )
        profile_name = _get_active_profile_name()
        st.markdown(
            f'<div class="sidebar-profile-name">{profile_name}</div>',
            unsafe_allow_html=True,
        )

        labels, ids, active_idx = _build_profile_options()
        if ids and ids[0]:
            selected_label = st.selectbox(
                "Switch profile",
                options=labels,
                index=active_idx,
                key="sidebar_profile_selector",
                label_visibility="collapsed",
            )
            if selected_label:
                sel_idx = labels.index(selected_label)
                selected_id = ids[sel_idx]
                current_active = st.session_state.get("profile_manager")
                if current_active is not None:
                    try:
                        existing_active_id = current_active.get_active_profile_id()
                        if existing_active_id != selected_id:
                            current_active.set_active_profile(selected_id)
                            logger.info("Switched profile to: %s", selected_id)
                            st.rerun()
                    except Exception as exc:
                        logger.error("Failed to switch profile: %s", exc)
                        st.error(f"Failed to switch profile: {exc}")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # -- Theme toggle --
        current_theme = st.session_state.get("theme", "dark")
        theme_label = "🌙 Dark" if current_theme == "dark" else "☀️ Light"
        if st.button(theme_label, key="theme_toggle", use_container_width=True):
            new_theme = "light" if current_theme == "dark" else "dark"
            st.session_state.theme = new_theme
            logger.info("Theme switched to: %s", new_theme)
            st.rerun()

        # -- Version footer --
        st.markdown(
            '<div class="sidebar-footer">Lullus v0.1.0</div>',
            unsafe_allow_html=True,
        )

    return st.session_state.current_page
