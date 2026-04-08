"""
Homework and text drafting module for UniMentor.

Generates academic drafts, improves student writing, and formats
output with proper citations from course materials.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class HomeworkDraft:
    """Represents a generated or improved homework draft."""
    content: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    format: str = "essay"
    word_count: int = 0


class HomeworkWriter:
    """Generates and improves academic homework drafts using RAG."""

    LENGTH_TARGETS = {
        "short": 500,
        "medium": 1000,
        "long": 2000,
    }

    FORMAT_DESCRIPTIONS = {
        "essay": "a structured academic essay with introduction, body paragraphs, and conclusion",
        "report": "a formal academic report with sections, headings, and findings",
        "summary": "a concise summary capturing the key points and takeaways",
        "analysis": "a critical analysis examining arguments, evidence, and implications",
    }

    def __init__(self, rag_engine, llm_engine) -> None:
        self.rag_engine = rag_engine
        self.llm_engine = llm_engine

    def generate_draft(
        self,
        topic: str,
        instructions: str,
        length: str,
        format: str,
        profile: dict,
    ) -> HomeworkDraft:
        """Generate a homework draft grounded in course materials.

        Args:
            topic: The topic to write about.
            instructions: Specific homework instructions or requirements.
            length: Target length — 'short', 'medium', or 'long'.
            format: Document format — 'essay', 'report', 'summary', or 'analysis'.
            profile: Student profile dict.

        Returns:
            HomeworkDraft with generated content, citations, and metadata.
        """
        target_words = self.LENGTH_TARGETS.get(length, 1000)
        format_desc = self.FORMAT_DESCRIPTIONS.get(format, self.FORMAT_DESCRIPTIONS["essay"])

        # Retrieve relevant context from course materials
        try:
            rag_response = self.rag_engine.query(topic, profile, mode="homework")
            context = rag_response.context_used
            sources = rag_response.sources
        except Exception as e:
            logger.warning("RAG retrieval failed, generating without context: %s", e)
            context = ""
            sources = []

        student_name = profile.get("student", {}).get("name", "Student")
        course_name = profile.get("course", {}).get("name", "the course")
        citation_style = profile.get("preferences", {}).get("citation_style", "APA")
        language = profile.get("course", {}).get("language", "english")
        tone = profile.get("preferences", {}).get("tone", "friendly_tutor")
        code_lang = profile.get("preferences", {}).get("code_language", "python")

        system_prompt = (
            f"You are an academic writing assistant helping {student_name} "
            f"with coursework for '{course_name}'. "
            f"Write in a clear, academic style appropriate for a "
            f"{profile.get('student', {}).get('degree_level', 'university')} student. "
            f"Use {citation_style} citation style when referencing materials. "
            f"Respond in {language}. "
            f"For any code examples, use {code_lang}."
        )

        user_prompt = (
            f"Write {format_desc} on the following topic:\n\n"
            f"**Topic:** {topic}\n\n"
            f"**Instructions:** {instructions}\n\n"
            f"**Target length:** approximately {target_words} words\n\n"
        )

        if context:
            user_prompt += (
                f"**Relevant course materials for reference:**\n"
                f"---\n{context}\n---\n\n"
                f"Use the above course materials to ground your writing. "
                f"Cite specific sources when drawing from them.\n"
            )

        user_prompt += (
            f"\nPlease produce the complete {format} with proper structure, "
            f"headings, and citations where appropriate."
        )

        try:
            content = self.llm_engine.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=max(2048, target_words * 3),
            )
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return HomeworkDraft(
                content=f"Error generating draft: {e}",
                citations=[],
                format=format,
                word_count=0,
            )

        citations = [
            {
                "filename": s.get("filename", "Unknown"),
                "page": str(s.get("page_number", "")),
                "section": s.get("section", ""),
                "relevance": str(s.get("relevance_score", "")),
            }
            for s in sources
        ]

        word_count = len(content.split())

        return HomeworkDraft(
            content=content,
            citations=citations,
            format=format,
            word_count=word_count,
        )

    def improve_draft(
        self,
        student_draft: str,
        instructions: str,
        profile: dict,
    ) -> HomeworkDraft:
        """Improve a student's existing draft with suggestions and edits.

        Args:
            student_draft: The student's current draft text.
            instructions: What kind of improvements the student wants.
            profile: Student profile dict.

        Returns:
            HomeworkDraft with the improved content.
        """
        student_name = profile.get("student", {}).get("name", "Student")
        course_name = profile.get("course", {}).get("name", "the course")
        citation_style = profile.get("preferences", {}).get("citation_style", "APA")
        language = profile.get("course", {}).get("language", "english")

        # Try to get relevant context for enrichment
        topic_hint = student_draft[:200]
        try:
            rag_response = self.rag_engine.query(topic_hint, profile, mode="homework")
            context = rag_response.context_used
            sources = rag_response.sources
        except Exception:
            context = ""
            sources = []

        system_prompt = (
            f"You are an academic writing tutor helping {student_name} improve their "
            f"coursework for '{course_name}'. "
            f"Provide constructive, specific feedback and produce an improved version. "
            f"Use {citation_style} citation style. Respond in {language}. "
            f"Be encouraging but honest about areas that need work."
        )

        user_prompt = (
            f"Please improve the following draft based on these instructions:\n\n"
            f"**Improvement instructions:** {instructions}\n\n"
            f"**Student's draft:**\n---\n{student_draft}\n---\n\n"
        )

        if context:
            user_prompt += (
                f"**Relevant course materials you can draw from:**\n"
                f"---\n{context}\n---\n\n"
            )

        user_prompt += (
            "Please provide:\n"
            "1. Brief feedback on the original draft (strengths and areas to improve)\n"
            "2. The improved version of the text\n"
        )

        try:
            content = self.llm_engine.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.6,
                max_tokens=4096,
            )
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return HomeworkDraft(
                content=f"Error improving draft: {e}",
                citations=[],
                format="essay",
                word_count=0,
            )

        citations = [
            {
                "filename": s.get("filename", "Unknown"),
                "page": str(s.get("page_number", "")),
                "section": s.get("section", ""),
            }
            for s in sources
        ]

        return HomeworkDraft(
            content=content,
            citations=citations,
            format="essay",
            word_count=len(content.split()),
        )
