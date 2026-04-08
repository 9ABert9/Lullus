"""
Full RAG (Retrieval-Augmented Generation) pipeline for Lullus.

Orchestrates the flow from query embedding through ChromaDB retrieval,
context assembly, prompt construction, and LLM generation. Supports
multiple modes (chat, exercise, research) and both blocking and
streaming generation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional  # noqa: F401

from app.core.defaults import DEFAULT_CONFIG
from app.prompts.system_prompts import (
    build_exercise_prompt,
    build_research_synthesis_prompt,
    build_system_prompt,
)

logger = logging.getLogger(__name__)

# Default similarity threshold -- documents below this score are discarded.
_DEFAULT_SIMILARITY_THRESHOLD: float = 0.35

# Maximum number of documents to retrieve from ChromaDB.
_DEFAULT_TOP_K: int = 8

# Maximum total characters of context to assemble (to stay within LLM limits).
_DEFAULT_MAX_CONTEXT_CHARS: int = 12_000


@dataclass
class RAGResponse:
    """Container for a RAG pipeline response.

    Attributes:
        answer: The generated answer text.
        sources: List of source metadata dicts, each containing filename,
            page_number, section, and relevance_score.
        confidence_score: A float between 0.0 and 1.0 indicating how
            confident the system is in the answer based on source quality.
        context_used: The assembled context string that was fed to the LLM.
    """

    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    context_used: str = ""


class RAGEngine:
    """Full RAG pipeline combining embedding, retrieval, and generation.

    Args:
        embedding_manager: An object with an ``embed_query(text: str) -> List[float]``
            method for producing query embeddings.
        llm_engine: An object with ``generate(prompt: str, system_prompt: str) -> str``
            and ``generate_stream(prompt: str, system_prompt: str) -> Generator[str, None, None]``
            methods.
        collection: A ChromaDB collection object supporting ``query()``.
        similarity_threshold: Minimum similarity score to keep a retrieved document.
        top_k: Number of documents to retrieve from ChromaDB.
        max_context_chars: Maximum total characters of assembled context.
    """

    def __init__(
        self,
        embedding_manager: Any,
        llm_engine: Any,
        collection: Any,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = _DEFAULT_TOP_K,
        max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        self._embedding_manager = embedding_manager
        self._llm_engine = llm_engine
        self._collection = collection
        self._similarity_threshold = similarity_threshold
        self._top_k = top_k
        self._max_context_chars = max_context_chars

        logger.info(
            "RAGEngine initialized (threshold=%.2f, top_k=%d, max_context=%d)",
            self._similarity_threshold,
            self._top_k,
            self._max_context_chars,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        profile: Optional[Dict[str, Any]] = None,
        mode: str = "chat",
        **kwargs: Any,
    ) -> RAGResponse:
        """Run the full RAG pipeline and return a complete response.

        Args:
            question: The user's question or instruction.
            profile: Optional config dict (defaults to DEFAULT_CONFIG).
            mode: One of 'chat', 'exercise', 'research'.
            **kwargs: Additional keyword arguments forwarded to the prompt
                builder (e.g., exercise_type, difficulty, topic).

        Returns:
            A RAGResponse with the answer, sources, confidence, and context.

        Raises:
            ValueError: If mode is not recognized.
        """
        profile = profile or DEFAULT_CONFIG
        self._validate_mode(mode)
        logger.info("RAG query started: mode=%s, question_len=%d", mode, len(question))

        # Step 1: Embed
        query_embedding = self._embed_query(question)

        # Step 2 & 3: Retrieve and filter
        documents, sources = self._retrieve_and_filter(query_embedding)

        # Step 4: Assemble context
        context = self._assemble_context(documents, sources)

        # Step 5: Build prompts
        system_prompt = self._build_system_prompt_for_mode(
            profile, mode, context, **kwargs
        )
        user_prompt = self._build_user_prompt(question, context, mode)

        # Step 6: Generate
        answer = self._generate(user_prompt, system_prompt)

        # Compute confidence
        confidence = self._compute_confidence(sources)

        logger.info(
            "RAG query complete: sources=%d, confidence=%.2f",
            len(sources),
            confidence,
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence_score=confidence,
            context_used=context,
        )

    def query_stream(
        self,
        question: str,
        profile: Optional[Dict[str, Any]] = None,
        mode: str = "chat",
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Run the RAG pipeline with streaming LLM output.

        Args:
            question: The user's question or instruction.
            profile: Optional config dict (defaults to DEFAULT_CONFIG).
            mode: One of 'chat', 'exercise', 'research'.
            **kwargs: Additional keyword arguments forwarded to the prompt builder.

        Yields:
            Partial answer strings (tokens or chunks).

        Raises:
            ValueError: If mode is not recognized.
        """
        profile = profile or DEFAULT_CONFIG
        self._validate_mode(mode)
        logger.info(
            "RAG streaming query started: mode=%s, question_len=%d",
            mode,
            len(question),
        )

        # Steps 1-4: same as non-streaming
        query_embedding = self._embed_query(question)
        documents, sources = self._retrieve_and_filter(query_embedding)
        context = self._assemble_context(documents, sources)

        # Step 5: Build prompts
        system_prompt = self._build_system_prompt_for_mode(
            profile, mode, context, **kwargs
        )
        user_prompt = self._build_user_prompt(question, context, mode)

        # Step 6: Stream generation
        logger.debug("Starting streaming generation")
        try:
            for chunk in self._llm_engine.generate_stream(
                prompt=user_prompt, system_prompt=system_prompt
            ):
                yield chunk
        except Exception:
            logger.exception("Error during streaming generation")
            raise

        logger.info("RAG streaming query complete: sources=%d", len(sources))

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_mode(mode: str) -> None:
        """Validate that the mode is supported.

        Args:
            mode: The requested pipeline mode.

        Raises:
            ValueError: If mode is not one of the valid options.
        """
        valid_modes = {"chat", "exercise", "research"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    def _embed_query(self, question: str) -> List[float]:
        """Embed the query text into a vector.

        Args:
            question: The raw question string.

        Returns:
            A list of floats representing the query embedding.
        """
        logger.debug("Embedding query (length=%d chars)", len(question))
        try:
            embedding = self._embedding_manager.embed_query(question)
            logger.debug("Query embedded successfully (dim=%d)", len(embedding))
            return embedding
        except Exception:
            logger.exception("Failed to embed query")
            raise

    def _retrieve_and_filter(
        self, query_embedding: List[float]
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Retrieve documents from ChromaDB and filter by similarity threshold.

        Args:
            query_embedding: The embedded query vector.

        Returns:
            A tuple of (filtered_documents, filtered_sources) where each source
            dict contains filename, page_number, section, and relevance_score.
        """
        logger.debug(
            "Querying ChromaDB: top_k=%d, threshold=%.2f",
            self._top_k,
            self._similarity_threshold,
        )
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=self._top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            logger.exception("ChromaDB query failed")
            raise

        raw_documents = results.get("documents", [[]])[0]
        raw_metadatas = results.get("metadatas", [[]])[0]
        raw_distances = results.get("distances", [[]])[0]

        filtered_documents: List[str] = []
        filtered_sources: List[Dict[str, Any]] = []

        for doc, meta, distance in zip(raw_documents, raw_metadatas, raw_distances):
            # ChromaDB returns L2 distances by default; convert to a
            # similarity-like score. For cosine distance collections the
            # distance is already between 0 and 2, so similarity = 1 - d/2.
            # For L2 we use a decay: similarity = 1 / (1 + distance).
            similarity = 1.0 / (1.0 + distance)

            if similarity < self._similarity_threshold:
                logger.debug(
                    "Filtered out document (similarity=%.3f < threshold=%.3f): %s",
                    similarity,
                    self._similarity_threshold,
                    meta.get("filename", "unknown"),
                )
                continue

            filtered_documents.append(doc)
            filtered_sources.append(
                {
                    "filename": meta.get("filename", "unknown"),
                    "page_number": meta.get("page_number"),
                    "section": meta.get("section", ""),
                    "relevance_score": round(similarity, 4),
                }
            )

        logger.info(
            "Retrieved %d documents, %d passed threshold",
            len(raw_documents),
            len(filtered_documents),
        )
        return filtered_documents, filtered_sources

    def _assemble_context(
        self, documents: List[str], sources: List[Dict[str, Any]]
    ) -> str:
        """Assemble retrieved documents into a single context string with source markers.

        Truncates to stay within max_context_chars.

        Args:
            documents: List of document text strings.
            sources: Corresponding source metadata dicts.

        Returns:
            A formatted context string with source attribution per chunk.
        """
        if not documents:
            logger.debug("No documents to assemble into context")
            return "No relevant course materials were found for this query."

        parts: List[str] = []
        total_chars = 0

        for i, (doc, src) in enumerate(zip(documents, sources)):
            page_info = (
                f", p. {src['page_number']}" if src.get("page_number") else ""
            )
            section_info = (
                f", section: {src['section']}" if src.get("section") else ""
            )
            header = (
                f"[Source {i + 1}: {src['filename']}{page_info}{section_info} "
                f"| relevance: {src['relevance_score']:.2f}]"
            )

            chunk = f"{header}\n{doc}"
            chunk_len = len(chunk)

            if total_chars + chunk_len > self._max_context_chars:
                remaining = self._max_context_chars - total_chars
                if remaining > 200:
                    # Include a truncated version of this chunk
                    parts.append(f"{header}\n{doc[:remaining - len(header) - 20]}...")
                    logger.debug(
                        "Truncated document %d to fit context limit", i + 1
                    )
                break

            parts.append(chunk)
            total_chars += chunk_len

        context = "\n\n---\n\n".join(parts)
        logger.debug("Assembled context: %d chars from %d documents", len(context), len(parts))
        return context

    def _build_system_prompt_for_mode(
        self,
        profile: Dict[str, Any],
        mode: str,
        context: str,
        **kwargs: Any,
    ) -> str:
        """Build the appropriate system prompt based on the pipeline mode.

        Args:
            profile: Student profile dictionary.
            mode: Pipeline mode ('chat', 'exercise', 'homework', 'research').
            context: The assembled RAG context string.
            **kwargs: Mode-specific parameters (exercise_type, difficulty, topic,
                instructions, length, formality).

        Returns:
            The system prompt string.
        """
        if mode == "chat":
            retrieval_mode = kwargs.get("retrieval_mode", "precise")
            return build_system_prompt(profile, retrieval_mode=retrieval_mode)

        if mode == "exercise":
            exercise_type = kwargs.get("exercise_type", "multiple_choice")
            difficulty = kwargs.get("difficulty", "medium")
            topic = kwargs.get("topic", "general")
            return build_exercise_prompt(profile, exercise_type, difficulty, topic)

        if mode == "research":
            topic = kwargs.get("topic", "general")
            return build_research_synthesis_prompt(profile, topic)

        return build_system_prompt(profile)

    @staticmethod
    def _build_user_prompt(question: str, context: str, mode: str) -> str:
        """Build the user-facing prompt that combines the question and context.

        Args:
            question: The user's original question.
            context: The assembled RAG context.
            mode: The pipeline mode (used for framing instructions).

        Returns:
            The user prompt string.
        """
        mode_instructions = {
            "chat": "Answer the following question using the course materials provided as context. Cite sources where applicable.",
            "exercise": "Using the following course materials as context, generate the requested exercise.",
            "research": "Synthesize the following research findings and course materials into a coherent overview.",
        }
        instruction = mode_instructions.get(mode, mode_instructions["chat"])

        return f"""{instruction}

## Retrieved Course Materials
{context}

## Student's Question / Request
{question}"""

    def _generate(self, user_prompt: str, system_prompt: str) -> str:
        """Generate a complete answer from the LLM.

        Args:
            user_prompt: The assembled user prompt with context and question.
            system_prompt: The system prompt defining AI behavior.

        Returns:
            The generated answer string.
        """
        logger.debug(
            "Generating response (system_prompt=%d chars, user_prompt=%d chars)",
            len(system_prompt),
            len(user_prompt),
        )
        try:
            answer = self._llm_engine.generate(
                prompt=user_prompt, system_prompt=system_prompt
            )
            logger.debug("Generation complete (%d chars)", len(answer))
            return answer
        except Exception:
            logger.exception("LLM generation failed")
            raise

    @staticmethod
    def _compute_confidence(sources: List[Dict[str, Any]]) -> float:
        """Compute a confidence score based on the quality of retrieved sources.

        The score is a weighted combination of:
        - The number of sources found (more sources = higher confidence, up to a point).
        - The average relevance score of the sources.

        Returns a float between 0.0 and 1.0.

        Args:
            sources: The list of source metadata dicts with relevance_score.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        if not sources:
            return 0.0

        avg_relevance = sum(s["relevance_score"] for s in sources) / len(sources)

        # Source count factor: peaks at 1.0 when there are 4+ good sources.
        count_factor = min(len(sources) / 4.0, 1.0)

        # Weighted combination: 70% relevance quality, 30% source coverage.
        confidence = 0.7 * avg_relevance + 0.3 * count_factor

        return round(min(max(confidence, 0.0), 1.0), 4)
