"""ChromaDB + Ollama embeddings manager for Lullus.

Manages document indexing, vector storage via ChromaDB, and semantic
search using Ollama embedding models. Includes discipline-specific
chunking strategies for Humanities/Social Sciences and STEM texts.
"""

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None  # type: ignore[assignment]

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore[assignment]

from app.core.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

COLLECTION_NAME = "lullus_documents"

# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

CHUNKING_STRATEGIES = {
    "humanities": {
        "label": "Humanities / Social Sciences",
        "description": (
            "Longer chunks (1200 chars) with 175-char overlap. "
            "Preserves paragraph boundaries, argumentative flow, "
            "block quotes, and discourse markers."
        ),
        "chunk_size": 1200,
        "chunk_overlap": 175,
    },
    "stem": {
        "label": "STEM / Technical",
        "description": (
            "Shorter precise chunks (384 chars) with 75-char overlap. "
            "Equation-aware splitting, preserves definitions, theorems, "
            "proofs, and numbered lists as atomic units."
        ),
        "chunk_size": 384,
        "chunk_overlap": 75,
    },
    "auto": {
        "label": "Auto (Balanced)",
        "description": (
            "Balanced default (512 chars, 50-char overlap). "
            "Good general-purpose strategy for mixed content."
        ),
        "chunk_size": 512,
        "chunk_overlap": 50,
    },
}


@dataclass
class DocumentInfo:
    """Metadata about an indexed document.

    Attributes:
        doc_id: Unique identifier for the document.
        filename: Original filename.
        chunk_count: Number of chunks the document was split into.
        file_size: File size in bytes.
        date_indexed: ISO timestamp of when the document was indexed.
        file_type: File extension (without dot).
    """

    doc_id: str
    filename: str
    chunk_count: int
    file_size: int
    date_indexed: str
    file_type: str


@dataclass
class ChunkResult:
    """A single search result chunk.

    Attributes:
        text: The chunk text content.
        source_file: Filename of the source document.
        page_number: Page number in the original document (1-based).
        section_title: Detected section title, if any.
        score: Similarity score (lower distance = more similar).
        metadata: Full metadata dictionary from ChromaDB.
    """

    text: str
    source_file: str
    page_number: int
    section_title: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class IndexingError(Exception):
    """Raised when document indexing fails."""


class EmbeddingManager:
    """Manage document embeddings with ChromaDB and Ollama.

    Stores document chunks as vectors in a persistent ChromaDB collection
    and uses Ollama's nomic-embed-text model for embedding generation.

    Args:
        base_dir: Root directory of the Lullus project.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        embedding_model: Ollama embedding model name.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        if chromadb is None:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        if ollama is None:
            raise ImportError(
                "ollama is required. Install with: pip install ollama"
            )

        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
        self.base_dir = Path(base_dir)

        # Load config
        config = self._load_config()
        emb_config: Dict[str, Any] = config.get("embeddings", {})

        self.chunk_size: int = chunk_size or emb_config.get("chunk_size", 512)
        self.chunk_overlap: int = chunk_overlap or emb_config.get("chunk_overlap", 50)
        self.embedding_model: str = embedding_model or emb_config.get("model", "nomic-embed-text")

        # Persistent ChromaDB storage
        self.chroma_dir = self.base_dir / "data" / "chroma_db"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self.doc_processor = DocumentProcessor()

        logger.info(
            "EmbeddingManager initialised: model=%s, chunk_size=%d, overlap=%d, "
            "chroma_dir=%s, collection_count=%d",
            self.embedding_model,
            self.chunk_size,
            self.chunk_overlap,
            self.chroma_dir,
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(
        self, file_path: str | Path, strategy: str = "auto"
    ) -> DocumentInfo:
        """Index a document: process, chunk, embed, and store.

        Args:
            file_path: Path to the document file.
            strategy: Chunking strategy — 'humanities', 'stem', or 'auto'.

        Returns:
            DocumentInfo with indexing metadata.

        Raises:
            IndexingError: If any step of the indexing pipeline fails.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if strategy not in CHUNKING_STRATEGIES:
            logger.warning("Unknown strategy '%s', falling back to 'auto'.", strategy)
            strategy = "auto"

        doc_id = self._generate_doc_id(path)

        # Remove existing chunks for this document (re-index case)
        self._remove_chunks_by_doc_id(doc_id)

        # Process the document
        logger.info(
            "Indexing document: %s (doc_id=%s, strategy=%s)",
            path.name, doc_id, strategy,
        )
        try:
            document = self.doc_processor.process(path)
        except Exception as exc:
            raise IndexingError(f"Document processing failed: {exc}") from exc

        # Chunk the document using the selected strategy
        strat = CHUNKING_STRATEGIES[strategy]
        chunks = self._chunk_document(
            document.pages,
            chunk_size=strat["chunk_size"],
            chunk_overlap=strat["chunk_overlap"],
            strategy=strategy,
        )
        if not chunks:
            raise IndexingError(f"No text chunks extracted from '{path.name}'")

        # Generate embeddings
        try:
            texts = [c["text"] for c in chunks]
            embeddings = self._embed_texts(texts)
        except Exception as exc:
            raise IndexingError(f"Embedding generation failed: {exc}") from exc

        # Build IDs and metadata for ChromaDB
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            ids.append(chunk_id)
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "filename": path.name,
                    "file_type": path.suffix.lstrip(".").lower(),
                    "page_number": chunk["page_number"],
                    "section_title": chunk["section_title"],
                    "chunk_index": idx,
                    "date_indexed": datetime.now().isoformat(),
                }
            )

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        info = DocumentInfo(
            doc_id=doc_id,
            filename=path.name,
            chunk_count=len(chunks),
            file_size=path.stat().st_size,
            date_indexed=datetime.now().isoformat(),
            file_type=path.suffix.lstrip(".").lower(),
        )
        logger.info(
            "Document indexed: %s (%d chunks)", path.name, len(chunks)
        )
        return info

    def remove_document(self, doc_id: str) -> None:
        """Remove all chunks for a document from the vector store.

        Args:
            doc_id: The document identifier.
        """
        removed = self._remove_chunks_by_doc_id(doc_id)
        logger.info("Removed document %s (%d chunks deleted)", doc_id, removed)

    def search(self, query: str, top_k: int = 5) -> List[ChunkResult]:
        """Perform semantic search across indexed documents.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.

        Returns:
            List of ChunkResult objects ordered by relevance.

        Raises:
            EmbeddingError: If embedding the query fails.
        """
        if self.collection.count() == 0:
            logger.warning("Search called on empty collection")
            return []

        try:
            query_embedding = self._embed_texts([query])[0]
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunk_results: List[ChunkResult] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for text, meta, distance in zip(documents, metadatas, distances):
            chunk_results.append(
                ChunkResult(
                    text=text or "",
                    source_file=meta.get("filename", "unknown"),
                    page_number=int(meta.get("page_number", 0)),
                    section_title=meta.get("section_title", ""),
                    score=float(distance),
                    metadata=dict(meta),
                )
            )

        logger.info("Search returned %d results for query (top_k=%d)", len(chunk_results), top_k)
        return chunk_results

    def get_all_documents(self) -> List[DocumentInfo]:
        """List all indexed documents with their metadata.

        Returns:
            List of DocumentInfo objects, one per unique document.
        """
        if self.collection.count() == 0:
            return []

        # Fetch all metadata to group by doc_id
        all_data = self.collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])

        doc_map: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            did = meta.get("doc_id", "unknown")
            if did not in doc_map:
                doc_map[did] = {
                    "doc_id": did,
                    "filename": meta.get("filename", "unknown"),
                    "file_type": meta.get("file_type", "unknown"),
                    "date_indexed": meta.get("date_indexed", ""),
                    "chunk_count": 0,
                }
            doc_map[did]["chunk_count"] += 1

        documents: List[DocumentInfo] = []
        for info in doc_map.values():
            documents.append(
                DocumentInfo(
                    doc_id=info["doc_id"],
                    filename=info["filename"],
                    chunk_count=info["chunk_count"],
                    file_size=0,  # Not stored per-chunk; would need the original file
                    date_indexed=info["date_indexed"],
                    file_type=info["file_type"],
                )
            )

        logger.debug("get_all_documents: %d documents found", len(documents))
        return documents

    def reindex_all(self, documents_dir: Optional[Path] = None) -> List[DocumentInfo]:
        """Clear the collection and re-index all documents in a directory.

        Args:
            documents_dir: Directory containing documents to index.
                Defaults to the knowledge_base folder.

        Returns:
            List of DocumentInfo for all successfully indexed documents.
        """
        if documents_dir is None:
            documents_dir = self.base_dir / "knowledge_base"

        if not documents_dir.exists():
            logger.warning("Documents directory does not exist: %s", documents_dir)
            return []

        # Clear existing collection
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared collection for reindex")

        from app.core.document_processor import SUPPORTED_FORMATS

        indexed: List[DocumentInfo] = []
        for file_path in sorted(documents_dir.rglob("*")):
            if file_path.suffix.lower() in SUPPORTED_FORMATS:
                try:
                    info = self.add_document(file_path)
                    indexed.append(info)
                except Exception as exc:
                    logger.error("Failed to index '%s': %s", file_path.name, exc)

        logger.info("Reindex complete: %d documents indexed", len(indexed))
        return indexed

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string into a vector.

        Args:
            text: The query text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        embeddings = self._embed_texts([text])
        return embeddings[0]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Ollama.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If the Ollama embedding call fails.
        """
        try:
            response = ollama.embed(model=self.embedding_model, input=texts)
            embeddings: List[List[float]] = response["embeddings"]
            logger.debug("Generated %d embeddings", len(embeddings))
            return embeddings
        except Exception as exc:
            error_msg = str(exc).lower()
            if "connection" in error_msg or "refused" in error_msg:
                raise EmbeddingError(
                    "Cannot connect to Ollama for embeddings. "
                    "Make sure Ollama is running with 'ollama serve' and "
                    f"the model '{self.embedding_model}' is pulled."
                ) from exc
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_document(
        self,
        pages: List[str],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: str = "auto",
    ) -> List[Dict[str, Any]]:
        """Split document pages into overlapping chunks.

        Applies discipline-specific splitting rules based on the strategy.

        Args:
            pages: List of page texts (one string per page).
            chunk_size: Override chunk size (chars).
            chunk_overlap: Override chunk overlap (chars).
            strategy: 'humanities', 'stem', or 'auto'.

        Returns:
            List of chunk dicts with keys: text, page_number,
            section_title.
        """
        cs = chunk_size or self.chunk_size
        co = chunk_overlap or self.chunk_overlap
        chunks: List[Dict[str, Any]] = []

        for page_idx, page_text in enumerate(pages):
            page_number = page_idx + 1
            sections = self._split_by_sections(page_text)

            for section_title, section_text in sections:
                if strategy == "humanities":
                    section_chunks = self._split_humanities(section_text, cs, co)
                elif strategy == "stem":
                    section_chunks = self._split_stem(section_text, cs, co)
                else:
                    section_chunks = self._split_into_chunks(section_text, cs, co)
                for chunk_text in section_chunks:
                    if chunk_text.strip():
                        chunks.append(
                            {
                                "text": chunk_text.strip(),
                                "page_number": page_number,
                                "section_title": section_title,
                            }
                        )

        logger.debug(
            "Chunked document into %d chunks (strategy=%s, chunk_size=%d)",
            len(chunks), strategy, cs,
        )
        return chunks

    def _split_by_sections(self, text: str) -> List[tuple]:
        """Split text into (section_title, section_body) pairs.

        Detects Markdown-style headers (# Title) and common heading
        patterns (ALL CAPS lines, numbered sections).

        Args:
            text: Full text of a page or chapter.

        Returns:
            List of (title, body) tuples.
        """
        # Pattern for Markdown headers and numbered sections
        header_pattern = re.compile(
            r"^(?:#{1,6}\s+(.+)|(\d+(?:\.\d+)*\.?\s+[A-Z].*)|([A-Z][A-Z\s]{5,}))$",
            re.MULTILINE,
        )

        matches = list(header_pattern.finditer(text))
        if not matches:
            return [("", text)]

        sections: List[tuple] = []

        # Text before first header
        pre_header = text[: matches[0].start()].strip()
        if pre_header:
            sections.append(("", pre_header))

        for i, match in enumerate(matches):
            title = (match.group(1) or match.group(2) or match.group(3) or "").strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body or title:
                sections.append((title, body))

        return sections if sections else [("", text)]

    def _split_into_chunks(
        self, text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """Split text into chunks of approximately chunk_size characters.

        Avoids breaking mid-sentence by finding the last sentence
        boundary within the chunk window.

        Args:
            text: Text to split.
            chunk_size: Override chunk size.
            chunk_overlap: Override chunk overlap.

        Returns:
            List of chunk strings.
        """
        cs = chunk_size or self.chunk_size
        co = chunk_overlap or self.chunk_overlap

        if len(text) <= cs:
            return [text] if text.strip() else []

        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + cs

            if end >= len(text):
                chunk = text[start:]
                if chunk.strip():
                    chunks.append(chunk)
                break

            # Try to find a sentence boundary near the end of the chunk
            boundary = self._find_sentence_boundary(text, start, end)
            chunk = text[start:boundary]
            if chunk.strip():
                chunks.append(chunk)

            # Move start with overlap
            start = max(boundary - co, start + 1)

        return chunks

    # ------------------------------------------------------------------
    # Humanities chunking
    # ------------------------------------------------------------------

    def _split_humanities(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Split text using paragraph-aware strategy for humanities texts.

        Keeps paragraphs intact when possible, preserves block quotes
        and discourse markers (however, therefore, in contrast, ...).
        Merges short paragraphs together up to chunk_size.

        Args:
            text: Section text.
            chunk_size: Target chunk size in chars.
            chunk_overlap: Overlap in chars.

        Returns:
            List of chunk strings.
        """
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        # Split on double-newline (paragraph boundary)
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text] if text.strip() else []

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)

            # If a single paragraph exceeds chunk_size, split it by sentences
            if para_len > chunk_size:
                # Flush current buffer first
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    # Keep last paragraph for overlap
                    overlap_text = current_chunk[-1] if current_chunk else ""
                    current_chunk = [overlap_text] if len(overlap_text) <= chunk_overlap else []
                    current_len = sum(len(c) for c in current_chunk)

                sub_chunks = self._split_into_chunks(para, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
                current_chunk = []
                current_len = 0
                continue

            # Check if adding this paragraph exceeds the limit
            new_len = current_len + para_len + (2 if current_chunk else 0)
            if new_len > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Overlap: keep last paragraph if it fits
                last = current_chunk[-1]
                if len(last) <= chunk_overlap:
                    current_chunk = [last]
                    current_len = len(last)
                else:
                    current_chunk = []
                    current_len = 0

            current_chunk.append(para)
            current_len += para_len + (2 if len(current_chunk) > 1 else 0)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    # ------------------------------------------------------------------
    # STEM chunking
    # ------------------------------------------------------------------

    # Patterns that mark atomic blocks we should never split
    _STEM_ATOMIC_PATTERNS = re.compile(
        r"(?:"
        r"(?:^|\n)(?:Definition|Theorem|Lemma|Corollary|Proposition|Proof|Remark|Example|Axiom|Postulate)"
        r"[\s.:]+.*?(?=\n(?:Definition|Theorem|Lemma|Corollary|Proposition|Proof|Remark|Example|Axiom|Postulate)[\s.:]|\Z)"
        r")",
        re.IGNORECASE | re.DOTALL,
    )

    # Equations: lines with $$...$$ or \[...\] or lines that are mostly math symbols
    _EQUATION_LINE = re.compile(
        r"^\s*(?:\$\$.*?\$\$|\\?\[.*?\\?\]|[=<>≤≥±∑∏∫√∞αβγδεζηθλμπσφψω\d\s\+\-\*/\^_{}()\[\]|,.:;]{10,})\s*$",
        re.MULTILINE,
    )

    def _split_stem(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Split text using STEM-aware strategy.

        Keeps equations, definitions, theorems, and numbered lists as
        atomic units. Uses shorter chunks for precise retrieval.

        Args:
            text: Section text.
            chunk_size: Target chunk size in chars.
            chunk_overlap: Overlap in chars.

        Returns:
            List of chunk strings.
        """
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        # First, try to extract atomic blocks (definitions, theorems, etc.)
        atomic_blocks = list(self._STEM_ATOMIC_PATTERNS.finditer(text))

        if not atomic_blocks:
            # No special STEM blocks — fall back to sentence-based splitting
            # but protect equation lines from being split
            return self._split_protecting_equations(text, chunk_size, chunk_overlap)

        chunks: List[str] = []
        last_end = 0

        for match in atomic_blocks:
            # Text before this block
            before = text[last_end:match.start()].strip()
            if before:
                sub = self._split_protecting_equations(before, chunk_size, chunk_overlap)
                chunks.extend(sub)

            block = match.group(0).strip()
            if len(block) <= chunk_size:
                chunks.append(block)
            else:
                # Oversized block — split by sentences but keep it labeled
                sub = self._split_into_chunks(block, chunk_size, chunk_overlap)
                chunks.extend(sub)

            last_end = match.end()

        # Remaining text after last block
        remaining = text[last_end:].strip()
        if remaining:
            sub = self._split_protecting_equations(remaining, chunk_size, chunk_overlap)
            chunks.extend(sub)

        return chunks

    def _split_protecting_equations(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Split text while keeping equation lines attached to surrounding text.

        If an equation line is near a sentence, it stays with that sentence.
        Falls back to normal sentence-boundary splitting.
        """
        # Simple approach: treat equation lines as paragraph separators
        # but keep them attached to the preceding text
        lines = text.split("\n")
        segments: List[str] = []
        current: List[str] = []

        for line in lines:
            current.append(line)
            joined = "\n".join(current)
            if len(joined) >= chunk_size:
                # Flush
                segments.append(joined)
                # Overlap: keep last few lines
                overlap_chars = 0
                overlap_lines: List[str] = []
                for prev_line in reversed(current):
                    overlap_chars += len(prev_line)
                    if overlap_chars > chunk_overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                current = overlap_lines

        if current:
            leftover = "\n".join(current).strip()
            if leftover:
                segments.append(leftover)

        # Further split any oversized segments
        final: List[str] = []
        for seg in segments:
            if len(seg) <= chunk_size:
                if seg.strip():
                    final.append(seg.strip())
            else:
                final.extend(self._split_into_chunks(seg, chunk_size, chunk_overlap))

        return final

    @staticmethod
    def _find_sentence_boundary(text: str, start: int, end: int) -> int:
        """Find the best sentence boundary position near end.

        Looks for sentence-ending punctuation followed by whitespace
        within the last 20% of the chunk. Falls back to the last
        whitespace if no sentence boundary is found.

        Args:
            text: Full text.
            start: Chunk start index.
            end: Target chunk end index.

        Returns:
            The best boundary position.
        """
        search_start = start + int((end - start) * 0.8)
        region = text[search_start:end]

        # Look for sentence boundaries (. ! ? followed by space or newline)
        sentence_end = None
        for match in re.finditer(r"[.!?]\s", region):
            sentence_end = search_start + match.end()

        if sentence_end is not None:
            return sentence_end

        # Fall back to last whitespace
        last_space = region.rfind(" ")
        if last_space != -1:
            return search_start + last_space + 1

        # Fall back to newline
        last_newline = region.rfind("\n")
        if last_newline != -1:
            return search_start + last_newline + 1

        return end

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_doc_id(path: Path) -> str:
        """Generate a deterministic document ID from the file path and size.

        Uses a SHA-256 hash of the absolute path and file size so that
        re-indexing the same file replaces old chunks.

        Args:
            path: Path to the document.

        Returns:
            A hex string document identifier.
        """
        key = f"{path.resolve()}:{path.stat().st_size}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def _remove_chunks_by_doc_id(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document.

        Args:
            doc_id: The document identifier.

        Returns:
            Number of chunks removed.
        """
        try:
            existing = self.collection.get(
                where={"doc_id": doc_id},
                include=[],
            )
            ids_to_remove = existing.get("ids", [])
            if ids_to_remove:
                self.collection.delete(ids=ids_to_remove)
                logger.debug("Removed %d existing chunks for doc_id=%s", len(ids_to_remove), doc_id)
            return len(ids_to_remove)
        except Exception as exc:
            logger.warning("Could not remove existing chunks for %s: %s", doc_id, exc)
            return 0

    def _load_config(self) -> Dict[str, Any]:
        """Load the default configuration YAML."""
        config_path = self.base_dir / "config" / "default_config.yaml"
        if not config_path.exists():
            logger.warning("Config not found at %s, using defaults", config_path)
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.warning("Error reading config: %s, using defaults", exc)
            return {}
