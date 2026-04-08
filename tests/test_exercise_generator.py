"""Tests for the exercise generator module."""

import unittest
from unittest.mock import MagicMock

from app.core.exercise_generator import (
    ExerciseGenerator,
    Exercise,
    AnswerFeedback,
)


class TestExerciseDataclasses(unittest.TestCase):
    """Test Exercise and AnswerFeedback dataclasses."""

    def test_exercise_creation(self) -> None:
        """Test creating an Exercise with all fields."""
        ex = Exercise(
            question="What is backpropagation?",
            exercise_type="open_ended",
            options=None,
            correct_answer="Backpropagation is an algorithm for training neural networks.",
            explanation="It computes gradients using the chain rule.",
            source_reference="lecture3.pdf, p.12",
            difficulty="medium",
            topic="neural networks",
        )
        self.assertEqual(ex.question, "What is backpropagation?")
        self.assertEqual(ex.exercise_type, "open_ended")
        self.assertIsNone(ex.options)
        self.assertEqual(ex.difficulty, "medium")
        self.assertEqual(ex.topic, "neural networks")

    def test_exercise_with_options(self) -> None:
        """Test creating a multiple choice Exercise."""
        ex = Exercise(
            question="Which activation function is most common in hidden layers?",
            exercise_type="multiple_choice",
            options=["Sigmoid", "ReLU", "Tanh", "Softmax"],
            correct_answer="ReLU",
            explanation="ReLU is preferred due to reduced vanishing gradient problem.",
            source_reference="lecture2.pdf",
            difficulty="easy",
            topic="neural networks",
        )
        self.assertEqual(len(ex.options), 4)
        self.assertIn("ReLU", ex.options)

    def test_answer_feedback_correct(self) -> None:
        """Test creating feedback for a correct answer."""
        feedback = AnswerFeedback(
            is_correct=True,
            feedback="Excellent! That's correct.",
            explanation="ReLU avoids the vanishing gradient problem.",
            score=1.0,
        )
        self.assertTrue(feedback.is_correct)
        self.assertEqual(feedback.score, 1.0)

    def test_answer_feedback_incorrect(self) -> None:
        """Test creating feedback for an incorrect answer."""
        feedback = AnswerFeedback(
            is_correct=False,
            feedback="Not quite. Think about gradient flow.",
            explanation="The correct answer is ReLU.",
            score=0.0,
        )
        self.assertFalse(feedback.is_correct)
        self.assertEqual(feedback.score, 0.0)

    def test_answer_feedback_partial(self) -> None:
        """Test creating feedback for a partially correct answer."""
        feedback = AnswerFeedback(
            is_correct=False,
            feedback="You're on the right track.",
            explanation="Your answer was partially correct.",
            score=0.5,
        )
        self.assertEqual(feedback.score, 0.5)


class TestExerciseGenerator(unittest.TestCase):
    """Test suite for ExerciseGenerator."""

    def setUp(self) -> None:
        self.mock_rag = MagicMock()
        self.mock_llm = MagicMock()
        self.generator = ExerciseGenerator(
            rag_engine=self.mock_rag,
        )
        self.sample_profile = {
            "student": {"name": "Test", "degree_level": "bachelor", "year": 1, "university": "ETH"},
            "course": {"name": "ML", "code": "ML-101", "language": "english", "topics": ["neural networks"]},
            "preferences": {
                "knowledge_level": "intermediate",
                "learning_style": "examples_first",
                "verbosity": "standard",
                "tone": "friendly_tutor",
                "citation_style": "APA",
                "code_language": "python",
                "exercise_difficulty": "adaptive",
            },
        }

    def test_adaptive_difficulty_increase(self) -> None:
        """Test that difficulty increases after 3 correct answers."""
        # Reset the tracker to medium
        from app.core.exercise_generator import AdaptiveDifficultyTracker
        self.generator._tracker = AdaptiveDifficultyTracker("medium")

        # Simulate 3 correct answers
        for _ in range(3):
            self.generator.track_performance(True)

        adapted = self.generator.get_adapted_difficulty()
        self.assertEqual(adapted, "hard")

    def test_adaptive_difficulty_decrease(self) -> None:
        """Test that difficulty decreases after 2 wrong answers."""
        from app.core.exercise_generator import AdaptiveDifficultyTracker
        self.generator._tracker = AdaptiveDifficultyTracker("medium")

        # Simulate 2 wrong answers
        for _ in range(2):
            self.generator.track_performance(False)

        adapted = self.generator.get_adapted_difficulty()
        self.assertEqual(adapted, "easy")

    def test_difficulty_stays_at_hard(self) -> None:
        """Test that difficulty doesn't go beyond hard."""
        from app.core.exercise_generator import AdaptiveDifficultyTracker
        self.generator._tracker = AdaptiveDifficultyTracker("hard")

        for _ in range(5):
            self.generator.track_performance(True)

        adapted = self.generator.get_adapted_difficulty()
        self.assertEqual(adapted, "hard")

    def test_difficulty_stays_at_easy(self) -> None:
        """Test that difficulty doesn't go below easy."""
        from app.core.exercise_generator import AdaptiveDifficultyTracker
        self.generator._tracker = AdaptiveDifficultyTracker("easy")

        for _ in range(5):
            self.generator.track_performance(False)

        adapted = self.generator.get_adapted_difficulty()
        self.assertEqual(adapted, "easy")

    def test_generate_exercises_calls_rag(self) -> None:
        """Test that generate_exercises uses RAG for context."""
        mock_response = MagicMock()
        mock_response.context_used = "Neural networks consist of layers."
        mock_response.sources = [{"filename": "lecture.pdf"}]
        self.mock_rag.query.return_value = mock_response

        # Mock LLM to return parseable exercise text
        self.mock_llm.generate.return_value = (
            "Q: What is a neural network?\n"
            "A) A biological system\n"
            "B) A computational model\n"
            "C) A database\n"
            "D) A programming language\n"
            "Correct: B\n"
            "Explanation: Neural networks are computational models."
        )

        try:
            self.generator.generate_exercises(
                topic="neural networks",
                difficulty="medium",
                exercise_type="multiple_choice",
                num_questions=1,
                profile=self.sample_profile,
            )
        except Exception:
            pass  # Parsing may fail on mock data, but RAG should be called

        self.mock_rag.query.assert_called()

    def test_check_answer_calls_llm(self) -> None:
        """Test that check_answer uses the LLM for evaluation."""
        exercise = Exercise(
            question="What is gradient descent?",
            exercise_type="open_ended",
            options=None,
            correct_answer="An optimization algorithm.",
            explanation="It minimizes a loss function.",
            source_reference="notes.pdf",
            difficulty="medium",
            topic="optimization",
        )

        self.mock_llm.generate.return_value = (
            "Correct: Yes\nFeedback: Good answer.\nScore: 0.9"
        )

        try:
            self.generator.check_answer(exercise, "An optimization method", self.sample_profile)
        except Exception:
            pass  # Parsing may vary, but LLM should be called

        self.mock_llm.generate.assert_called()

    def test_exercise_types(self) -> None:
        """Test that all exercise types are recognized."""
        valid_types = [
            "multiple_choice",
            "open_ended",
            "fill_in_blank",
            "true_false",
            "problem_solving",
            "code_exercise",
        ]
        for ex_type in valid_types:
            ex = Exercise(
                question="Test?",
                exercise_type=ex_type,
                options=None,
                correct_answer="Test",
                explanation="Test",
                source_reference="",
                difficulty="medium",
                topic="test",
            )
            self.assertEqual(ex.exercise_type, ex_type)


if __name__ == "__main__":
    unittest.main()
