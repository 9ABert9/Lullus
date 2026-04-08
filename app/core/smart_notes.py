"""
Smart Notes module for Lullus.

Takes rough notes, keywords, or broken sentences from the user and
produces polished, readable text enriched with relevant information
from the knowledge base.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from app.core.defaults import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class EnhancedNotes:
    """Result of the smart notes enhancement."""
    enhanced_text: str
    sources_used: List[Dict[str, str]] = field(default_factory=list)
    word_count: int = 0
    topics_detected: List[str] = field(default_factory=list)


class SmartNotes:
    """Enhances rough notes using RAG to produce polished, complete text."""

    def __init__(self, rag_engine, llm_engine) -> None:
        self.rag_engine = rag_engine
        self.llm_engine = llm_engine

    def enhance_notes(
        self,
        raw_notes: str,
        style: str = "detailed",
        language: str = "",
        citation_style: str = "",
    ) -> EnhancedNotes:
        """Transform rough notes into polished, enriched text.

        Takes raw input (keywords, fragments, broken sentences) and:
        1. Generates a readable, well-structured text from the notes
        2. Integrates relevant information from the knowledge base
        3. Completes gaps where information is missing
        4. Adds references to source materials

        Args:
            raw_notes: The rough notes, keywords, or fragments.
            style: Output style — 'concise', 'detailed', or 'outline'.
            language: Response language override. Falls back to DEFAULT_CONFIG.
            citation_style: Citation style override.

        Returns:
            EnhancedNotes with the polished text, sources, and metadata.
        """
        language = language or DEFAULT_CONFIG.get("language", "english")
        citation_style = citation_style or DEFAULT_CONFIG.get("citation_style", "APA")

        # Retrieve relevant context from knowledge base.
        context, sources = self._multi_query_rag(raw_notes)

        # Estimate target length
        input_words = len(raw_notes.split())
        expansion = {"concise": 2.0, "detailed": 3.5, "outline": 2.5}
        target_words = max(500, int(input_words * expansion.get(style, 3.0)))

        style_instructions = {
            "concise": (
                f"Write a clear, well-structured text of approximately {target_words} words. "
                f"Cover every single point from the notes. Be efficient but do not skip anything."
            ),
            "detailed": (
                f"Write a thorough, comprehensive text of approximately {target_words} words or more. "
                f"Elaborate on EVERY point from the notes in depth. For each concept or topic mentioned, "
                f"provide a full explanation, context, and connections to related ideas. "
                f"Use paragraphs with clear topic sentences and smooth transitions. "
                f"Do NOT summarize — EXPAND and ELABORATE."
            ),
            "outline": (
                f"Create a comprehensive structured outline of approximately {target_words} words. "
                f"Use headings, sub-headings, and detailed bullet points. "
                f"Each point from the notes must appear with a full explanation underneath it."
            ),
        }

        style_desc = style_instructions.get(style, style_instructions["detailed"])

        system_prompt = (
            f"You are an expert note-enhancement assistant. Your task is to transform "
            f"rough, incomplete notes into a COMPLETE, polished, well-structured document.\n\n"
            f"CRITICAL RULES:\n"
            f"1. Process EVERY SINGLE point, keyword, and fragment in the notes — do not skip or merge items.\n"
            f"2. ELABORATE each point into a full explanation. A keyword like 'gradient descent' should become "
            f"a full paragraph explaining what it is, how it works, and why it matters.\n"
            f"3. Look up relevant information in the provided course materials and INTEGRATE it: "
            f"explain concepts in depth, add definitions, provide context, and connect ideas.\n"
            f"4. Where the notes are incomplete or vague, FILL IN the missing information "
            f"using the course materials. Explain what the user likely meant and expand on it.\n"
            f"5. Mark information added from course materials with [Source: filename] at the end "
            f"of the relevant paragraph.\n"
            f"6. The output must be SIGNIFICANTLY LONGER and more detailed than the input notes.\n"
            f"7. Use proper academic language and structure (sections, paragraphs, transitions).\n"
            f"8. If a concept is mentioned but not in the course materials, explain it using "
            f"general knowledge and mark it as [General knowledge].\n"
            f"9. Use {citation_style} style for references.\n"
            f"10. Respond in {language}.\n\n"
            f"Output style: {style_desc}"
        )

        user_prompt = (
            f"Transform the following rough notes into a comprehensive, elaborated document. "
            f"Every point must be fully developed. The output should be approximately "
            f"{target_words} words — much longer and richer than the input.\n\n"
            f"--- MY ROUGH NOTES ---\n{raw_notes}\n--- END NOTES ---\n\n"
        )

        if context:
            user_prompt += (
                f"--- RELEVANT COURSE MATERIALS (use these to explain and enrich) ---\n"
                f"{context}\n"
                f"--- END MATERIALS ---\n\n"
                f"INSTRUCTIONS: Use the course materials above to:\n"
                f"- Explain every concept mentioned in my notes\n"
                f"- Add relevant context, definitions, and background\n"
                f"- Fill in gaps where my notes are incomplete\n"
                f"- Connect ideas to the broader course content\n"
                f"- Cite sources with [Source: filename]\n"
            )
        else:
            user_prompt += (
                "No course materials were found. Please enhance the notes using "
                "general knowledge, explain all concepts thoroughly, and indicate "
                "with [General knowledge] where you are drawing from outside the course."
            )

        max_tokens = max(4096, min(8192, target_words * 3))

        try:
            enhanced_text = self.llm_engine.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return EnhancedNotes(
                enhanced_text=f"Error enhancing notes: {e}",
                sources_used=[],
                word_count=0,
                topics_detected=[],
            )

        citations = [
            {
                "filename": s.get("filename", "Unknown"),
                "page": str(s.get("page_number", "")),
                "section": s.get("section", ""),
                "relevance": str(round(s.get("relevance_score", 0), 2)),
            }
            for s in sources
        ]

        return EnhancedNotes(
            enhanced_text=enhanced_text,
            sources_used=citations,
            word_count=len(enhanced_text.split()),
            topics_detected=[],
        )

    def _multi_query_rag(self, raw_notes: str) -> tuple:
        """Run multiple RAG queries to cover different parts of the notes.

        Splits the notes into chunks by line groups and queries each
        separately, then merges all retrieved context and sources.

        Args:
            raw_notes: The raw notes text.

        Returns:
            Tuple of (combined_context: str, all_sources: list).
        """
        lines = [l.strip() for l in raw_notes.strip().split("\n") if l.strip()]
        segments: List[str] = []

        if len(lines) <= 5:
            segments = [raw_notes]
        else:
            chunk_size = 5
            for i in range(0, len(lines), chunk_size):
                segment = "\n".join(lines[i : i + chunk_size])
                if segment.strip():
                    segments.append(segment)

        if len(segments) > 1:
            segments.append(raw_notes)

        all_context_parts: List[str] = []
        all_sources: List[Dict] = []
        seen_filenames: set = set()

        for segment in segments:
            try:
                rag_response = self.rag_engine.query(segment, mode="chat")
                if rag_response.context_used:
                    all_context_parts.append(rag_response.context_used)
                for src in rag_response.sources:
                    key = (
                        src.get("filename", ""),
                        src.get("page_number", ""),
                    )
                    if key not in seen_filenames:
                        seen_filenames.add(key)
                        all_sources.append(src)
            except Exception as e:
                logger.warning("RAG query failed for segment: %s", e)

        combined = "\n\n---\n\n".join(all_context_parts) if all_context_parts else ""

        logger.info(
            "Multi-query RAG: %d segments queried, %d unique sources, %d chars context",
            len(segments), len(all_sources), len(combined),
        )

        return combined, all_sources
