"""Tests for the RAG engine module."""

import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from app.core.rag_engine import RAGEngine, RAGResponse


class TestRAGResponse(unittest.TestCase):
    """Test the RAGResponse dataclass."""

    def test_create_rag_response(self) -> None:
        """Test creating a RAGResponse with all fields."""
        response = RAGResponse(
            answer="This is the answer.",
            sources=[{"filename": "notes.pdf", "page_number": 5, "relevance_score": 0.92}],
            confidence_score=0.85,
            context_used="Some context from documents.",
        )
        self.assertEqual(response.answer, "This is the answer.")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.confidence_score, 0.85)
        self.assertEqual(response.context_used, "Some context from documents.")

    def test_empty_sources(self) -> None:
        """Test RAGResponse with empty sources."""
        response = RAGResponse(
            answer="General knowledge answer.",
            sources=[],
            confidence_score=0.3,
            context_used="",
        )
        self.assertEqual(response.sources, [])
        self.assertEqual(response.confidence_score, 0.3)


class TestRAGEngine(unittest.TestCase):
    """Test suite for RAGEngine."""

    def setUp(self) -> None:
        self.mock_embedding_manager = MagicMock()
        self.mock_llm_engine = MagicMock()
        self.engine = RAGEngine(
            embedding_manager=self.mock_embedding_manager,
            llm_engine=self.mock_llm_engine,
            top_k=3,
            similarity_threshold=0.3,
        )
        self.sample_profile = {
            "student": {"name": "Test Student", "degree_level": "master", "year": 1, "university": "ETH"},
            "course": {"name": "ML", "code": "ML-101", "language": "english", "topics": ["neural networks"]},
            "preferences": {
                "knowledge_level": "intermediate",
                "learning_style": "examples_first",
                "verbosity": "standard",
                "tone": "friendly_tutor",
                "citation_style": "APA",
                "code_language": "python",
            },
        }

    def test_query_with_results(self) -> None:
        """Test query when embedding search returns results."""
        # Mock search results
        mock_chunk = MagicMock()
        mock_chunk.text = "Neural networks are computational models."
        mock_chunk.source_file = "lecture1.pdf"
        mock_chunk.page_number = 3
        mock_chunk.section_title = "Introduction"
        mock_chunk.score = 0.88
        mock_chunk.metadata = {}

        self.mock_embedding_manager.search.return_value = [mock_chunk]
        self.mock_llm_engine.generate.return_value = "Neural networks are inspired by biological neurons."

        result = self.engine.query("What are neural networks?", self.sample_profile, mode="chat")

        self.assertIsInstance(result, RAGResponse)
        self.assertIn("neural networks", result.answer.lower())
        self.mock_embedding_manager.search.assert_called_once()
        self.mock_llm_engine.generate.assert_called_once()

    def test_query_empty_knowledge_base(self) -> None:
        """Test query when no documents are indexed."""
        self.mock_embedding_manager.search.return_value = []
        self.mock_llm_engine.generate.return_value = "I don't have course materials on this topic."

        result = self.engine.query("What is gradient descent?", self.sample_profile, mode="chat")

        self.assertIsInstance(result, RAGResponse)
        self.assertEqual(len(result.sources), 0)

    def test_query_modes(self) -> None:
        """Test that different modes are accepted."""
        self.mock_embedding_manager.search.return_value = []
        self.mock_llm_engine.generate.return_value = "Response text."

        for mode in ["chat", "exercise", "homework", "research"]:
            result = self.engine.query("Test question", self.sample_profile, mode=mode)
            self.assertIsInstance(result, RAGResponse)

    def test_source_tracking(self) -> None:
        """Test that sources are properly tracked in the response."""
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "Content from lecture 1."
        mock_chunk1.source_file = "lecture1.pdf"
        mock_chunk1.page_number = 1
        mock_chunk1.section_title = "Intro"
        mock_chunk1.score = 0.9
        mock_chunk1.metadata = {}

        mock_chunk2 = MagicMock()
        mock_chunk2.text = "Content from lecture 2."
        mock_chunk2.source_file = "lecture2.pdf"
        mock_chunk2.page_number = 5
        mock_chunk2.section_title = "Methods"
        mock_chunk2.score = 0.7
        mock_chunk2.metadata = {}

        self.mock_embedding_manager.search.return_value = [mock_chunk1, mock_chunk2]
        self.mock_llm_engine.generate.return_value = "Combined answer."

        result = self.engine.query("Explain methods", self.sample_profile, mode="chat")

        self.assertEqual(len(result.sources), 2)

    def test_context_assembly(self) -> None:
        """Test that context is assembled from multiple chunks."""
        mock_chunk = MagicMock()
        mock_chunk.text = "Important content."
        mock_chunk.source_file = "notes.pdf"
        mock_chunk.page_number = 1
        mock_chunk.section_title = "Section 1"
        mock_chunk.score = 0.85
        mock_chunk.metadata = {}

        self.mock_embedding_manager.search.return_value = [mock_chunk]
        self.mock_llm_engine.generate.return_value = "Answer using context."

        result = self.engine.query("Test", self.sample_profile, mode="chat")

        self.assertTrue(len(result.context_used) > 0)

    def test_llm_failure_handling(self) -> None:
        """Test graceful handling when LLM fails."""
        self.mock_embedding_manager.search.return_value = []
        self.mock_llm_engine.generate.side_effect = Exception("Ollama not running")

        with self.assertRaises(Exception):
            self.engine.query("Test", self.sample_profile, mode="chat")


if __name__ == "__main__":
    unittest.main()
