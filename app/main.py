"""
Lullus — Your Local AI Knowledge Organizer & Smart Notes

Main Streamlit application entry point. Handles initialization,
routing, and page layout.
"""

import logging
import sys
from pathlib import Path

import streamlit as st
import yaml

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.llm_engine import LLMEngine
from app.core.embedding_manager import EmbeddingManager
from app.core.rag_engine import RAGEngine
from app.core.document_processor import DocumentProcessor
from app.core.exercise_generator import ExerciseGenerator
from app.core.knowledge_checker import KnowledgeChecker
from app.core.web_researcher import WebResearcher
from app.core.smart_notes import SmartNotes
from app.core.defaults import DEFAULT_CONFIG
from app.utils.file_watcher import FileWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(
    page_title="Lullus",
    page_icon="\U0001F4DA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Global dark theme enhancements */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }

    /* Card styling */
    .card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Status indicators */
    .status-online { color: #3fb950; }
    .status-offline { color: #f85149; }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 600;
    }

    /* Expander */
    .streamlit-expanderHeader {
        border-radius: 8px;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        border: 2px dashed #444;
        border-radius: 12px;
        padding: 16px;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: #667eea;
    }

    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 8px;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Score display */
    .score-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Configuration ---
def load_config() -> dict:
    """Load the application configuration from YAML."""
    config_path = PROJECT_ROOT / "config" / "user_config.yaml"
    if not config_path.exists():
        config_path = PROJECT_ROOT / "config" / "default_config.yaml"
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return {}


# --- Initialize Components ---
def init_components() -> None:
    """Initialize all core components in session state."""
    if "initialized" in st.session_state:
        return

    config = load_config()
    st.session_state.config = config

    # LLM Engine
    llm_config = config.get("llm", {})
    llm_engine = LLMEngine(
        base_dir=PROJECT_ROOT,
        model=llm_config.get("model"),
        temperature=llm_config.get("temperature"),
        max_tokens=llm_config.get("max_tokens"),
    )
    st.session_state.llm_engine = llm_engine

    # Document Processor
    doc_processor = DocumentProcessor()
    st.session_state.document_processor = doc_processor

    # Embedding Manager
    emb_config = config.get("embeddings", {})
    embedding_manager = EmbeddingManager(
        base_dir=PROJECT_ROOT,
        embedding_model=emb_config.get("model"),
        chunk_size=emb_config.get("chunk_size"),
        chunk_overlap=emb_config.get("chunk_overlap"),
    )
    st.session_state.embedding_manager = embedding_manager

    # RAG Engine
    rag_config = config.get("rag", {})
    rag_engine = RAGEngine(
        embedding_manager=embedding_manager,
        llm_engine=llm_engine,
        collection=embedding_manager.collection,
        top_k=rag_config.get("top_k", 5),
        similarity_threshold=rag_config.get("similarity_threshold", 0.3),
    )
    st.session_state.rag_engine = rag_engine

    # Exercise Generator
    exercise_generator = ExerciseGenerator(
        rag_engine=rag_engine,
        model=llm_engine.model,
    )
    st.session_state.exercise_generator = exercise_generator

    # Knowledge Checker
    knowledge_checker = KnowledgeChecker(
        rag_engine=rag_engine,
        model=llm_engine.model,
    )
    st.session_state.knowledge_checker = knowledge_checker

    # Smart Notes
    smart_notes = SmartNotes(rag_engine=rag_engine, llm_engine=llm_engine)
    st.session_state.smart_notes = smart_notes

    # Web Researcher
    web_config = config.get("web_search", {})
    web_researcher = WebResearcher(
        llm_engine=llm_engine,
        max_results=web_config.get("max_results", 5),
    )
    st.session_state.web_researcher = web_researcher

    # File Watcher
    kb_config = config.get("knowledge_base", {})
    watch_folder = kb_config.get("watch_folder", "./knowledge_base")
    file_watcher = FileWatcher(
        watch_path=str(PROJECT_ROOT / watch_folder.lstrip("./")),
        embedding_manager=embedding_manager,
        document_processor=doc_processor,
    )
    st.session_state.file_watcher = file_watcher

    # Auto-start file watcher if configured
    if kb_config.get("auto_index", True):
        try:
            file_watcher.start()
        except Exception as e:
            logger.warning("Could not start file watcher: %s", e)

    st.session_state.initialized = True
    logger.info("All components initialized")


# --- Sidebar ---
def render_sidebar() -> str:
    """Render the navigation sidebar and return the selected page."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 10px 0 20px 0;">
                <h2 style="margin: 0;">\U0001F4DA Lullus</h2>
                <p style="color: #888; font-size: 0.85em; margin: 4px 0 0 0;">
                    Knowledge Organizer & Smart Notes
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Ollama connection status
        llm_engine = st.session_state.get("llm_engine")
        if llm_engine:
            connected = llm_engine.check_connection()
            if connected:
                st.markdown(
                    '<span class="status-online">\u25CF</span> Ollama connected',
                    unsafe_allow_html=True,
                )
                st.caption(f"Model: {llm_engine.model}")
            else:
                st.markdown(
                    '<span class="status-offline">\u25CF</span> Ollama disconnected',
                    unsafe_allow_html=True,
                )
                st.caption("Start Ollama to use the assistant")

        st.markdown("---")

        # Navigation
        pages = {
            "\U0001F4AC Chat": "chat",
            "\U0001F4DA Knowledge Base": "knowledge",
            "\u2728 Smart Notes": "smart_notes",
            "\U0001F3AF Knowledge Assessment": "assessment",
            "\u2699\uFE0F Settings": "settings",
        }

        if "current_page" not in st.session_state:
            st.session_state.current_page = "chat"

        for label, page_key in pages.items():
            if st.button(label, use_container_width=True, key=f"nav_{page_key}"):
                st.session_state.current_page = page_key
                st.rerun()

        st.markdown("---")

        # Stats
        embedding_manager = st.session_state.get("embedding_manager")
        if embedding_manager:
            try:
                docs = embedding_manager.get_all_documents()
                st.metric("Indexed Documents", len(docs))
            except Exception:
                st.metric("Indexed Documents", 0)

    return st.session_state.current_page


# --- Settings Page ---
def render_settings() -> None:
    """Render the settings page."""
    st.markdown("## \u2699\uFE0F Settings")

    config = st.session_state.get("config", {})
    llm_engine = st.session_state.get("llm_engine")

    st.markdown("### LLM Configuration")

    # Model selector
    llm_config = config.get("llm", {})
    all_models = [llm_config.get("model", "mistral:7b-instruct-v0.3-q4_K_M")]
    all_models.extend(llm_config.get("alternative_models", []))

    if llm_engine:
        try:
            available = llm_engine.list_models()
            for m in available:
                if m not in all_models:
                    all_models.append(m)
        except Exception:
            pass

    selected_model = st.selectbox(
        "Model",
        all_models,
        index=0,
        key="settings_model",
    )

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=llm_config.get("temperature", 0.7),
            step=0.1,
            key="settings_temp",
        )
    with col2:
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=256,
            max_value=8192,
            value=llm_config.get("max_tokens", 2048),
            step=256,
            key="settings_tokens",
        )

    st.markdown("### RAG Configuration")
    rag_config = config.get("rag", {})
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=20,
            value=rag_config.get("top_k", 5),
            key="settings_topk",
        )
    with col2:
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=rag_config.get("similarity_threshold", 0.3),
            step=0.05,
            key="settings_threshold",
        )

    st.markdown("### Web Search")
    web_config = config.get("web_search", {})
    web_enabled = st.checkbox(
        "Enable Web Search",
        value=web_config.get("enabled", True),
        key="settings_web",
    )

    if st.button("Save Settings", type="primary"):
        config["llm"]["model"] = selected_model
        config["llm"]["temperature"] = temperature
        config["llm"]["max_tokens"] = max_tokens
        config["rag"]["top_k"] = top_k
        config["rag"]["similarity_threshold"] = threshold
        config["web_search"]["enabled"] = web_enabled

        config_path = PROJECT_ROOT / "config" / "user_config.yaml"
        try:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            st.success("Settings saved!")

            # Update runtime components
            if llm_engine:
                llm_engine.model = selected_model
                llm_engine.temperature = temperature
                llm_engine.max_tokens = max_tokens
            rag_engine = st.session_state.get("rag_engine")
            if rag_engine:
                rag_engine.top_k = top_k
                rag_engine.similarity_threshold = threshold
        except Exception as e:
            st.error(f"Failed to save settings: {e}")


# --- Main App ---
def main() -> None:
    """Main application entry point."""
    init_components()

    # Render sidebar and get current page
    current_page = render_sidebar()

    # Route to the correct page
    if current_page == "chat":
        from app.ui.chat_panel import render_chat_panel
        render_chat_panel()

    elif current_page == "knowledge":
        from app.ui.knowledge_manager import render_knowledge_manager
        render_knowledge_manager(
            embedding_manager=st.session_state.embedding_manager,
            file_watcher=st.session_state.file_watcher,
        )

    elif current_page == "smart_notes":
        from app.ui.smart_notes_panel import render_smart_notes
        render_smart_notes(
            smart_notes_engine=st.session_state.smart_notes,
        )

    elif current_page == "assessment":
        from app.ui.exercise_panel import render_knowledge_assessment
        render_knowledge_assessment(
            exercise_generator=st.session_state.exercise_generator,
            knowledge_checker=st.session_state.knowledge_checker,
        )

    elif current_page == "settings":
        render_settings()


if __name__ == "__main__":
    main()
