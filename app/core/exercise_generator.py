"""
Exercise generation engine for Lullus.

Generates various types of exercises grounded in course material using RAG,
with adaptive difficulty tracking based on student performance.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import ollama

from app.core.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Exercise:
    """A single exercise/question."""
    question: str
    exercise_type: str
    options: Optional[List[str]]
    correct_answer: str
    explanation: str
    source_reference: str
    difficulty: str
    topic: str


@dataclass
class AnswerFeedback:
    """Feedback returned after checking a student's answer."""
    is_correct: bool
    feedback: str
    explanation: str
    score: float  # 0.0 - 1.0


# ---------------------------------------------------------------------------
# Prompt templates (exercise_templates replacement)
# ---------------------------------------------------------------------------

EXERCISE_SYSTEM_PROMPT = (
    "You are a university-level exercise generator. "
    "Create exercises that are pedagogically sound, clear, and grounded in the "
    "provided course material. Always output valid JSON."
)

EXERCISE_TYPE_TEMPLATES: Dict[str, str] = {
    "multiple_choice": (
        "Generate {num} {difficulty} multiple-choice questions about '{topic}'.\n"
        "Each question must have exactly 4 options (A, B, C, D) with one correct answer.\n"
        "Use the following course material as the basis:\n---\n{context}\n---\n"
        "Student profile: {profile_summary}\n\n"
        "Return a JSON array where each element has:\n"
        '  "question": string,\n'
        '  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
        '  "correct_answer": "A" | "B" | "C" | "D",\n'
        '  "explanation": string,\n'
        '  "source_reference": string\n'
    ),
    "open_ended": (
        "Generate {num} {difficulty} open-ended questions about '{topic}'.\n"
        "Use the following course material as the basis:\n---\n{context}\n---\n"
        "Student profile: {profile_summary}\n\n"
        "Return a JSON array where each element has:\n"
        '  "question": string,\n'
        '  "correct_answer": string (model answer),\n'
        '  "explanation": string,\n'
        '  "source_reference": string\n'
    ),
    "fill_in_blank": (
        "Generate {num} {difficulty} fill-in-the-blank sentences about '{topic}'.\n"
        "Mark the blank with '______'.\n"
        "Use the following course material as the basis:\n---\n{context}\n---\n"
        "Student profile: {profile_summary}\n\n"
        "Return a JSON array where each element has:\n"
        '  "question": string (sentence with blank),\n'
        '  "correct_answer": string (word/phrase for the blank),\n'
        '  "explanation": string,\n'
        '  "source_reference": string\n'
    ),
    "true_false": (
        "Generate {num} {difficulty} true/false statements about '{topic}'.\n"
        "Use the following course material as the basis:\n---\n{context}\n---\n"
        "Student profile: {profile_summary}\n\n"
        "Return a JSON array where each element has:\n"
        '  "question": string (statement),\n'
        '  "correct_answer": "True" | "False",\n'
        '  "explanation": string,\n'
        '  "source_reference": string\n'
    ),
    "problem_solving": (
        "Generate {num} {difficulty} problem-solving exercises about '{topic}'.\n"
        "Each problem should require multi-step reasoning.\n"
        "Use the following course material as the basis:\n---\n{context}\n---\n"
        "Student profile: {profile_summary}\n\n"
        "Return a JSON array where each element has:\n"
        '  "question": string,\n'
        '  "correct_answer": string (full worked solution),\n'
        '  "explanation": string,\n'
        '  "source_reference": string\n'
    ),
    "code_exercise": (
        "Generate {num} {difficulty} coding exercises about '{topic}'.\n"
        "Preferred language: {code_language}.\n"
        "Use the following course material as the basis:\n---\n{context}\n---\n"
        "Student profile: {profile_summary}\n\n"
        "Return a JSON array where each element has:\n"
        '  "question": string (problem description),\n'
        '  "correct_answer": string (complete solution code),\n'
        '  "explanation": string,\n'
        '  "source_reference": string\n'
    ),
}

CHECK_ANSWER_PROMPT = (
    "You are grading a student's answer.\n"
    "Question: {question}\n"
    "Expected answer: {correct_answer}\n"
    "Student answer: {student_answer}\n"
    "Exercise type: {exercise_type}\n\n"
    "Evaluate correctness and provide constructive feedback.\n"
    "Return JSON with:\n"
    '  "is_correct": boolean,\n'
    '  "score": float (0.0 to 1.0),\n'
    '  "feedback": string (encouraging, specific),\n'
    '  "explanation": string (why the answer is right/wrong)\n'
)


# ---------------------------------------------------------------------------
# Adaptive difficulty tracker
# ---------------------------------------------------------------------------

class AdaptiveDifficultyTracker:
    """Tracks student performance and adapts difficulty level.

    Increases difficulty after 3 consecutive correct answers.
    Decreases difficulty after 2 consecutive wrong answers.
    """

    LEVELS = ["easy", "medium", "hard"]

    def __init__(self, starting_level: str = "medium") -> None:
        if starting_level not in self.LEVELS:
            starting_level = "medium"
        self._level_index: int = self.LEVELS.index(starting_level)
        self._consecutive_correct: int = 0
        self._consecutive_wrong: int = 0

    def track_performance(self, is_correct: bool) -> None:
        """Record whether the student answered correctly and adjust counters."""
        if is_correct:
            self._consecutive_correct += 1
            self._consecutive_wrong = 0
            if self._consecutive_correct >= 3:
                self._level_index = min(self._level_index + 1, len(self.LEVELS) - 1)
                self._consecutive_correct = 0
                logger.info("Adaptive difficulty increased to %s", self.get_adapted_difficulty())
        else:
            self._consecutive_wrong += 1
            self._consecutive_correct = 0
            if self._consecutive_wrong >= 2:
                self._level_index = max(self._level_index - 1, 0)
                self._consecutive_wrong = 0
                logger.info("Adaptive difficulty decreased to %s", self.get_adapted_difficulty())

    def get_adapted_difficulty(self) -> str:
        """Return the current adapted difficulty level."""
        return self.LEVELS[self._level_index]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class ExerciseGenerator:
    """Generates exercises grounded in course material via RAG.

    Parameters
    ----------
    rag_engine : RAGEngine
        A configured RAG engine for retrieving course context.
    model : str
        Ollama model name to use for generation.
    """

    def __init__(self, rag_engine: RAGEngine, model: str = "mistral:7b-instruct-v0.3-q4_K_M") -> None:
        self._rag = rag_engine
        self._model = model
        self._tracker = AdaptiveDifficultyTracker()

    # ------------------------------------------------------------------
    # Public helpers for adaptive tracking
    # ------------------------------------------------------------------

    def track_performance(self, is_correct: bool) -> None:
        """Delegate performance tracking to the adaptive tracker."""
        self._tracker.track_performance(is_correct)

    def get_adapted_difficulty(self) -> str:
        """Return the current adapted difficulty from the tracker."""
        return self._tracker.get_adapted_difficulty()

    # ------------------------------------------------------------------
    # Exercise generation
    # ------------------------------------------------------------------

    def generate_exercises(
        self,
        topic: str,
        difficulty: str = "medium",
        exercise_type: str = "multiple_choice",
        num_questions: int = 5,
    ) -> List[Exercise]:
        """Generate a list of exercises for the given topic.

        Parameters
        ----------
        topic : str
            The subject/topic for the exercises.
        difficulty : str
            One of ``easy``, ``medium``, ``hard``, or ``adaptive``.
        exercise_type : str
            One of the keys in ``EXERCISE_TYPE_TEMPLATES``.
        num_questions : int
            How many exercises to produce.

        Returns
        -------
        List[Exercise]
            A list of generated Exercise objects.
        """
        if exercise_type not in EXERCISE_TYPE_TEMPLATES:
            logger.error("Unknown exercise type '%s'. Falling back to multiple_choice.", exercise_type)
            exercise_type = "multiple_choice"

        if difficulty == "adaptive":
            difficulty = self._tracker.get_adapted_difficulty()

        from app.core.defaults import DEFAULT_CONFIG
        profile_summary = f"Knowledge level: {DEFAULT_CONFIG.get('knowledge_level', 'intermediate')}"
        code_language = DEFAULT_CONFIG.get("code_language", "python")

        # Retrieve relevant context via RAG
        context = self._retrieve_context(topic)

        prompt_template = EXERCISE_TYPE_TEMPLATES[exercise_type]
        user_prompt = prompt_template.format(
            num=num_questions,
            difficulty=difficulty,
            topic=topic,
            context=context,
            profile_summary=profile_summary,
            code_language=code_language,
        )

        raw_response = self._call_llm(user_prompt)
        exercises = self._parse_exercises(raw_response, exercise_type, difficulty, topic)

        if not exercises:
            logger.warning("LLM returned no parseable exercises; returning empty list.")
        else:
            logger.info("Generated %d %s exercises on '%s' (difficulty=%s).", len(exercises), exercise_type, topic, difficulty)

        return exercises

    # ------------------------------------------------------------------
    # Answer checking
    # ------------------------------------------------------------------

    def check_answer(
        self,
        exercise: Exercise,
        student_answer: str,
    ) -> AnswerFeedback:
        """Check a student answer against the exercise and return feedback.

        Parameters
        ----------
        exercise : Exercise
            The exercise that was answered.
        student_answer : str
            The student's submitted answer.
        profile : dict, optional
            Student profile for tone adjustment.

        Returns
        -------
        AnswerFeedback
        """
        prompt = CHECK_ANSWER_PROMPT.format(
            question=exercise.question,
            correct_answer=exercise.correct_answer,
            student_answer=student_answer,
            exercise_type=exercise.exercise_type,
        )

        raw = self._call_llm(prompt)
        feedback = self._parse_feedback(raw, exercise, student_answer)

        # Feed result into adaptive tracker
        self._tracker.track_performance(feedback.is_correct)

        return feedback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve_context(self, topic: str) -> str:
        """Use RAG to get relevant course material for the topic."""
        try:
            response = self._rag.query(topic, mode="exercise")
            if response.context_used:
                return response.context_used
            logger.warning("RAG returned no context for topic '%s'.", topic)
            return "(No specific course material found for this topic.)"
        except Exception:
            logger.exception("Error retrieving RAG context for topic '%s'.", topic)
            return "(Could not retrieve course material.)"

    def _call_llm(self, user_prompt: str) -> str:
        """Send a prompt to the Ollama LLM and return the response text."""
        try:
            response = ollama.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": EXERCISE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.7},
            )
            return response["message"]["content"]
        except Exception:
            logger.exception("LLM call failed.")
            return "[]"

    @staticmethod
    def _parse_exercises(
        raw: str,
        exercise_type: str,
        difficulty: str,
        topic: str,
    ) -> List[Exercise]:
        """Parse JSON response from the LLM into Exercise objects."""
        import json

        # Try to extract a JSON array from the response
        raw = raw.strip()
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            logger.error("Could not find JSON array in LLM response.")
            return []

        try:
            items = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            logger.exception("Failed to parse LLM response as JSON.")
            return []

        exercises: List[Exercise] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            exercises.append(
                Exercise(
                    question=item.get("question", ""),
                    exercise_type=exercise_type,
                    options=item.get("options"),
                    correct_answer=str(item.get("correct_answer", "")),
                    explanation=item.get("explanation", ""),
                    source_reference=item.get("source_reference", ""),
                    difficulty=difficulty,
                    topic=topic,
                )
            )
        return exercises

    @staticmethod
    def _parse_feedback(raw: str, exercise: Exercise, student_answer: str) -> AnswerFeedback:
        """Parse the LLM grading response into an AnswerFeedback object."""
        import json

        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                data = json.loads(raw[start:end + 1])
                return AnswerFeedback(
                    is_correct=bool(data.get("is_correct", False)),
                    feedback=data.get("feedback", ""),
                    explanation=data.get("explanation", ""),
                    score=float(data.get("score", 0.0)),
                )
            except (json.JSONDecodeError, ValueError):
                logger.exception("Failed to parse feedback JSON.")

        # Fallback: simple string comparison
        normalised_student = student_answer.strip().lower()
        normalised_correct = exercise.correct_answer.strip().lower()
        is_correct = normalised_student == normalised_correct
        return AnswerFeedback(
            is_correct=is_correct,
            feedback="Correct!" if is_correct else "Incorrect. Please review the explanation.",
            explanation=exercise.explanation,
            score=1.0 if is_correct else 0.0,
        )
