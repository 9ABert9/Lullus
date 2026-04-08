"""
Knowledge assessment module for Lullus.

Provides adaptive knowledge checks that adjust difficulty based on student
performance, generates assessment reports, and persists history to disk.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama

from app.core.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

# Path for persisting assessment history
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
HISTORY_FILE = DATA_DIR / "assessment_history.json"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AssessmentReport:
    """Final report produced at the end of a knowledge assessment."""
    topic: str
    total_questions: int
    correct_count: int
    score_percentage: float
    topics_understood: List[str]
    topics_to_review: List[str]
    recommended_materials: List[str]
    difficulty_progression: List[str]
    user_name: str = ""
    timestamp: float = 0.0


@dataclass
class _QuestionRecord:
    """Internal record of a single question in the assessment."""
    question: str
    correct_answer: str
    student_answer: str = ""
    is_correct: bool = False
    difficulty: str = "medium"
    sub_topic: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a university-level knowledge assessor. Generate clear, focused "
    "questions that test understanding of the topic. Always output valid JSON."
)

QUESTION_PROMPT = (
    "Generate 1 {difficulty} question about '{topic}' to assess a student's knowledge.\n"
    "Course material:\n---\n{context}\n---\n"
    "Student profile: {profile_summary}\n\n"
    "Return JSON with:\n"
    '  "question": string,\n'
    '  "correct_answer": string,\n'
    '  "sub_topic": string (specific sub-topic tested)\n'
)

EVALUATE_PROMPT = (
    "Evaluate this student answer.\n"
    "Question: {question}\n"
    "Expected: {correct_answer}\n"
    "Student answer: {student_answer}\n\n"
    "Return JSON with:\n"
    '  "is_correct": boolean,\n'
    '  "explanation": string\n'
)

REPORT_PROMPT = (
    "Based on these assessment results for the topic '{topic}', provide analysis.\n"
    "Results:\n{results_json}\n\n"
    "Return JSON with:\n"
    '  "topics_understood": [list of sub-topics the student knows well],\n'
    '  "topics_to_review": [list of sub-topics needing more study],\n'
    '  "recommended_materials": [list of source files or references to review]\n'
)

# ---------------------------------------------------------------------------
# Difficulty helpers
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


def _adjust_difficulty(current: str, correct: bool) -> str:
    """Simple one-step adjustment: correct -> harder, wrong -> easier."""
    idx = DIFFICULTY_LEVELS.index(current) if current in DIFFICULTY_LEVELS else 1
    if correct:
        idx = min(idx + 1, len(DIFFICULTY_LEVELS) - 1)
    else:
        idx = max(idx - 1, 0)
    return DIFFICULTY_LEVELS[idx]


# ---------------------------------------------------------------------------
# KnowledgeChecker
# ---------------------------------------------------------------------------

class KnowledgeChecker:
    """Conducts adaptive knowledge assessments.

    Usage
    -----
    >>> checker = KnowledgeChecker(rag_engine)
    >>> first_q = checker.start_assessment("neural networks", profile)
    >>> result = checker.submit_answer("Backpropagation")
    >>> # result is either the next question (str) or an AssessmentReport
    """

    DEFAULT_NUM_QUESTIONS = 5

    def __init__(
        self,
        rag_engine: RAGEngine,
        model: str = "mistral:7b-instruct-v0.3-q4_K_M",
        num_questions: int = 5,
    ) -> None:
        self._rag = rag_engine
        self._model = model
        self._num_questions = max(1, num_questions)
        self._topic: str = ""
        self._profile: Dict[str, Any] = {}
        self._records: List[_QuestionRecord] = []
        self._current_difficulty: str = "medium"
        self._current_question: Optional[_QuestionRecord] = None
        self._finished: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_assessment(
        self,
        topic: str,
        profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Begin a new assessment session and return the first question.

        Parameters
        ----------
        topic : str
            The topic to assess.
        profile : dict, optional
            Student profile dictionary.

        Returns
        -------
        str
            The text of the first question.
        """
        self._topic = topic
        self._profile = profile or {}
        self._records = []
        self._current_difficulty = "medium"
        self._finished = False

        logger.info("Starting assessment on '%s' (%d questions).", topic, self._num_questions)

        question_text = self._generate_question()
        return question_text

    def submit_answer(self, answer: str) -> "str | AssessmentReport":
        """Submit an answer to the current question.

        Returns the next question as a string, or an ``AssessmentReport`` if
        the assessment is complete.

        Parameters
        ----------
        answer : str
            The student's answer.

        Returns
        -------
        str or AssessmentReport
        """
        if self._finished:
            logger.warning("Assessment already finished.")
            return self._generate_report()

        if self._current_question is None:
            logger.error("No current question. Call start_assessment first.")
            return "Error: assessment not started."

        # Evaluate the answer
        is_correct = self._evaluate_answer(self._current_question, answer)
        self._current_question.student_answer = answer
        self._current_question.is_correct = is_correct
        self._records.append(self._current_question)
        self._current_question = None

        # Adjust difficulty
        self._current_difficulty = _adjust_difficulty(self._current_difficulty, is_correct)

        # Check if done
        if len(self._records) >= self._num_questions:
            self._finished = True
            report = self._generate_report()
            self._save_report(report)
            return report

        # Generate next question
        next_question = self._generate_question()
        return next_question

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    @staticmethod
    def load_history() -> List[Dict[str, Any]]:
        """Load all past assessment reports.

        Returns
        -------
        list of dict
            List of past assessment report dictionaries.
        """
        if not HISTORY_FILE.exists():
            return []
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to read assessment history file.")
            return []

        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_question(self) -> str:
        """Use RAG + LLM to generate a single assessment question."""
        context = self._retrieve_context(self._topic)
        profile_summary = self._build_profile_summary(self._profile)

        prompt = QUESTION_PROMPT.format(
            difficulty=self._current_difficulty,
            topic=self._topic,
            context=context,
            profile_summary=profile_summary,
        )

        raw = self._call_llm(prompt)
        parsed = self._parse_json(raw)

        question_text = parsed.get("question", f"Explain a key concept in {self._topic}.")
        correct_answer = parsed.get("correct_answer", "")
        sub_topic = parsed.get("sub_topic", self._topic)

        self._current_question = _QuestionRecord(
            question=question_text,
            correct_answer=correct_answer,
            difficulty=self._current_difficulty,
            sub_topic=sub_topic,
        )

        return question_text

    def _evaluate_answer(self, record: _QuestionRecord, student_answer: str) -> bool:
        """Use the LLM to evaluate whether the student's answer is correct."""
        prompt = EVALUATE_PROMPT.format(
            question=record.question,
            correct_answer=record.correct_answer,
            student_answer=student_answer,
        )
        raw = self._call_llm(prompt)
        parsed = self._parse_json(raw)
        return bool(parsed.get("is_correct", False))

    def _generate_report(self) -> AssessmentReport:
        """Compile an AssessmentReport from recorded question results."""
        correct_count = sum(1 for r in self._records if r.is_correct)
        total = len(self._records)
        score_pct = (correct_count / total * 100.0) if total > 0 else 0.0
        difficulty_progression = [r.difficulty for r in self._records]

        # Ask LLM for topic analysis
        results_json = json.dumps(
            [
                {
                    "sub_topic": r.sub_topic,
                    "difficulty": r.difficulty,
                    "is_correct": r.is_correct,
                }
                for r in self._records
            ],
            indent=2,
        )
        prompt = REPORT_PROMPT.format(topic=self._topic, results_json=results_json)
        raw = self._call_llm(prompt)
        analysis = self._parse_json(raw)

        return AssessmentReport(
            topic=self._topic,
            total_questions=total,
            correct_count=correct_count,
            score_percentage=round(score_pct, 1),
            topics_understood=analysis.get("topics_understood", []),
            topics_to_review=analysis.get("topics_to_review", []),
            recommended_materials=analysis.get("recommended_materials", []),
            difficulty_progression=difficulty_progression,
            user_name="user",
            timestamp=time.time(),
        )

    def _save_report(self, report: AssessmentReport) -> None:
        """Append the report to the assessment history JSON file."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        history: List[Dict[str, Any]] = []
        if HISTORY_FILE.exists():
            try:
                history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not read existing history; starting fresh.")

        history.append(asdict(report))

        try:
            HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Assessment report saved to %s.", HISTORY_FILE)
        except OSError:
            logger.exception("Failed to save assessment history.")

    def _retrieve_context(self, topic: str) -> str:
        """Retrieve relevant course material from RAG."""
        try:
            response = self._rag.query(topic, mode="chat")
            if response.context_used:
                return response.context_used
            return "(No specific course material found.)"
        except Exception:
            logger.exception("RAG query failed for '%s'.", topic)
            return "(Could not retrieve course material.)"

    @staticmethod
    def _build_profile_summary(profile: Dict[str, Any]) -> str:
        """Create a brief text summary from config."""
        if not profile:
            return "No profile available."
        parts = []
        if profile.get("knowledge_level"):
            parts.append(f"Knowledge: {profile['knowledge_level']}")
        if profile.get("language"):
            parts.append(f"Language: {profile['language']}")
        return "; ".join(parts) if parts else "General user."

    def _call_llm(self, user_prompt: str) -> str:
        """Call the Ollama LLM and return the response text."""
        try:
            response = ollama.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.7},
            )
            return response["message"]["content"]
        except Exception:
            logger.exception("LLM call failed.")
            return "{}"

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """Extract the first JSON object or array from a raw LLM response."""
        raw = raw.strip()
        # Try object
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        # Try array (return first element if present)
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            try:
                arr = json.loads(raw[start:end + 1])
                if arr and isinstance(arr[0], dict):
                    return arr[0]
            except json.JSONDecodeError:
                pass
        logger.warning("Could not parse JSON from LLM response.")
        return {}
