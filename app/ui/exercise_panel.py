"""
Knowledge Assessment UI for Lullus.

Unified panel combining exercise generation and adaptive knowledge checks,
all grounded in the Knowledge Base documents.
"""

import logging
from typing import List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

EXERCISE_TYPES = [
    "Multiple Choice",
    "Open-ended",
    "Fill-in-the-blank",
    "True/False",
    "Problem-solving",
    "Code Exercise",
]

EXERCISE_TYPE_MAP = {
    "Multiple Choice": "multiple_choice",
    "Open-ended": "open_ended",
    "Fill-in-the-blank": "fill_in_blank",
    "True/False": "true_false",
    "Problem-solving": "problem_solving",
    "Code Exercise": "code_exercise",
}

DIFFICULTY_MAP = {0: "easy", 1: "medium", 2: "hard"}


def _init_state() -> None:
    """Initialize all assessment session state variables."""
    defaults = {
        "exercises": [],
        "exercise_answers": {},
        "exercise_feedback": {},
        "exercise_score": {"correct": 0, "total": 0},
        "exercises_generated": False,
        "kc_active": False,
        "kc_questions": [],
        "kc_current": 0,
        "kc_answers": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_knowledge_assessment(
    exercise_generator,
    knowledge_checker,
) -> None:
    """Render the unified Knowledge Assessment page.

    Args:
        exercise_generator: ExerciseGenerator instance.
        knowledge_checker: KnowledgeChecker instance.
    """
    _init_state()

    st.markdown("## \U0001F3AF Knowledge Assessment")
    st.info(
        "All questions and assessments are **generated from your Knowledge Base**. "
        "The more documents you upload, the richer and more relevant the exercises will be.",
        icon="\U0001F4DA",
    )

    # Check indexed doc count
    embedding_manager = st.session_state.get("embedding_manager")
    if embedding_manager:
        try:
            doc_count = len(embedding_manager.get_all_documents())
        except Exception:
            doc_count = 0
        if doc_count == 0:
            st.warning(
                "Your Knowledge Base is empty. Go to **Knowledge Base** and upload "
                "course materials first to get relevant exercises and assessments."
            )

    tab_exercises, tab_assessment, tab_history = st.tabs([
        "\U0001F9EA Practice Exercises",
        "\U0001F4CB Adaptive Assessment",
        "\U0001F4CA History",
    ])

    with tab_exercises:
        _render_exercises_tab(exercise_generator)

    with tab_assessment:
        _render_assessment_tab(knowledge_checker)

    with tab_history:
        _render_history_tab(knowledge_checker)


# ---------------------------------------------------------------------------
# Tab 1: Practice Exercises
# ---------------------------------------------------------------------------

def _render_exercises_tab(exercise_generator) -> None:
    """Render the exercise generation and quiz tab."""

    st.markdown("### Generate Practice Exercises")
    st.caption("Create exercises from your course materials to test specific topics.")

    col1, col2 = st.columns(2)

    with col1:
        topic = st.text_input(
            "Topic",
            placeholder="e.g., Neural Networks, Gradient Descent...",
            key="exercise_topic",
        )
        exercise_type = st.radio(
            "Exercise Type",
            EXERCISE_TYPES,
            key="exercise_type_select",
            horizontal=True,
        )

    with col2:
        difficulty_idx = st.slider(
            "Difficulty",
            min_value=0,
            max_value=2,
            value=1,
            format="",
            key="exercise_difficulty",
        )
        difficulty_labels = {0: "\U0001F7E2 Easy", 1: "\U0001F7E1 Medium", 2: "\U0001F534 Hard"}
        st.markdown(f"**Level:** {difficulty_labels[difficulty_idx]}")

        num_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=20,
            value=5,
            key="exercise_num_questions",
        )

    st.markdown("---")

    col_gen, col_clear = st.columns([3, 1])
    with col_gen:
        generate_clicked = st.button(
            "\U0001F680 Generate Exercises",
            use_container_width=True,
            type="primary",
            disabled=not topic,
        )
    with col_clear:
        if st.button("\U0001F5D1\uFE0F Clear All", use_container_width=True, key="clear_exercises"):
            st.session_state.exercises = []
            st.session_state.exercise_answers = {}
            st.session_state.exercise_feedback = {}
            st.session_state.exercise_score = {"correct": 0, "total": 0}
            st.session_state.exercises_generated = False
            st.rerun()

    if generate_clicked and not topic:
        st.warning("Please enter a topic first.")
        return

    if generate_clicked and topic:
        difficulty = DIFFICULTY_MAP[difficulty_idx]
        ex_type = EXERCISE_TYPE_MAP[exercise_type]

        with st.spinner(f"Generating {num_questions} exercises on '{topic}' from your Knowledge Base..."):
            try:
                exercises = exercise_generator.generate_exercises(
                    topic=topic,
                    difficulty=difficulty,
                    exercise_type=ex_type,
                    num_questions=num_questions,
                )
                st.session_state.exercises = exercises
                st.session_state.exercise_answers = {}
                st.session_state.exercise_feedback = {}
                st.session_state.exercise_score = {"correct": 0, "total": 0}
                st.session_state.exercises_generated = True
            except Exception as e:
                st.error(f"Failed to generate exercises: {e}")
                logger.error("Exercise generation failed: %s", e)
                return

    # Display exercises
    exercises = st.session_state.exercises
    if not exercises:
        if not st.session_state.exercises_generated:
            st.markdown(
                """
                <div style="text-align: center; padding: 40px 20px;
                            border: 2px dashed #555; border-radius: 12px;
                            margin: 20px 0;">
                    <p style="font-size: 1.2em;">\U0001F3AF Ready to practice?</p>
                    <p style="color: #888;">Choose a topic and settings above, then click Generate.
                    Questions are created from your uploaded materials.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # Score
    score = st.session_state.exercise_score
    if score["total"] > 0:
        pct = int(score["correct"] / score["total"] * 100)
        st.markdown(f"### \U0001F4CA Score: {score['correct']}/{score['total']} ({pct}%)")
        st.progress(score["correct"] / max(len(exercises), 1))

    st.markdown("---")

    for i, ex in enumerate(exercises):
        _render_single_exercise(i, ex, exercise_generator)

    # Export
    if exercises:
        from app.utils.export_utils import export_exercises_to_markdown
        md_content = export_exercises_to_markdown(exercises, title=f"Exercises: {topic}")
        st.download_button(
            "\U0001F4E5 Export Exercises (Markdown)",
            data=md_content,
            file_name=f"exercises_{topic.replace(' ', '_') if topic else 'exercises'}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 2: Adaptive Assessment
# ---------------------------------------------------------------------------

def _render_assessment_tab(knowledge_checker) -> None:
    """Render the adaptive knowledge assessment tab."""

    st.markdown("### Adaptive Knowledge Assessment")
    st.caption(
        "Test your understanding with adaptive questions. "
        "Difficulty adjusts based on your answers. All questions come from your Knowledge Base."
    )

    if not st.session_state.kc_active:
        topic = st.text_input(
            "Topic to assess",
            placeholder="e.g., EOSC Task Forces, Neural Networks...",
            key="kc_topic_input",
        )
        num_questions = st.slider(
            "Number of questions",
            5, 10, 5,
            key="kc_num_input",
        )

        if st.button("\u25B6\uFE0F Start Assessment", type="primary", disabled=not topic, key="start_assessment"):
            with st.spinner("Preparing assessment from your Knowledge Base..."):
                try:
                    first_q = knowledge_checker.start_assessment(topic)
                    st.session_state.kc_active = True
                    st.session_state.kc_active_topic = topic
                    st.session_state.kc_questions = [first_q]
                    st.session_state.kc_current = 0
                    st.session_state.kc_answers = []
                    st.session_state.kc_report = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start assessment: {e}")

    else:
        current = st.session_state.kc_current
        questions = st.session_state.kc_questions
        topic = st.session_state.get("kc_active_topic", "")

        st.markdown(f"**Topic:** {topic}")

        total_expected = max(len(questions), current + 1)
        st.progress(current / total_expected)
        st.caption(f"Question {current + 1}")

        if current < len(questions):
            q = questions[current]
            q_text = q if isinstance(q, str) else q.get("question", str(q))

            st.markdown(f"#### {q_text}")

            answer = st.text_area(
                "Your answer:",
                key=f"kc_answer_{current}",
                placeholder="Type your answer here...",
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("Submit Answer", type="primary", key=f"kc_submit_{current}"):
                    if answer.strip():
                        with st.spinner("Evaluating your answer..."):
                            try:
                                result = knowledge_checker.submit_answer(answer)
                                st.session_state.kc_answers.append({
                                    "question": q_text,
                                    "answer": answer,
                                    "result": result,
                                })
                                if isinstance(result, str):
                                    st.session_state.kc_questions.append(result)
                                else:
                                    st.session_state.kc_report = result
                                st.session_state.kc_current += 1
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.warning("Please write an answer first.")
            with col2:
                if st.button("Cancel", key="kc_cancel"):
                    st.session_state.kc_active = False
                    st.session_state.kc_questions = []
                    st.session_state.kc_current = 0
                    st.session_state.kc_answers = []
                    st.session_state.kc_report = None
                    st.rerun()
        else:
            st.markdown("### \u2705 Assessment Complete!")
            report = st.session_state.get("kc_report")
            if report:
                score = getattr(report, "score_percentage", 0)
                st.markdown(f"**Overall Score: {score}%**")
                st.progress(score / 100)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Topics understood well:**")
                    for t in getattr(report, "topics_understood", []):
                        st.markdown(f"- \u2705 {t}")
                with col2:
                    st.markdown("**Topics to review:**")
                    for t in getattr(report, "topics_to_review", []):
                        st.markdown(f"- \u26A0\uFE0F {t}")

                recs = getattr(report, "recommended_materials", [])
                if recs:
                    st.markdown("**Recommended materials to revisit:**")
                    for r in recs:
                        st.markdown(f"- \U0001F4DA {r}")
            else:
                st.info("Assessment recorded.")

            if st.button("Start New Assessment", key="kc_new"):
                st.session_state.kc_active = False
                st.session_state.kc_questions = []
                st.session_state.kc_current = 0
                st.session_state.kc_answers = []
                st.session_state.kc_report = None
                st.rerun()


# ---------------------------------------------------------------------------
# Tab 3: History
# ---------------------------------------------------------------------------

def _render_history_tab(knowledge_checker) -> None:
    """Render past assessment history."""
    st.markdown("### Past Assessments")

    history = knowledge_checker.load_history()
    if not history:
        st.caption("No assessments completed yet. Start one in the Adaptive Assessment tab!")
        return

    for entry in reversed(history[-10:]):
        topic = entry.get("topic", "Unknown")
        date = entry.get("date", "")
        score = entry.get("score_percentage", 0)
        total = entry.get("total_questions", 0)

        with st.expander(f"**{topic}** \u2014 {score}% ({date})"):
            st.markdown(f"**Questions:** {total}")
            st.markdown(f"**Score:** {score}%")
            st.progress(score / 100)

            understood = entry.get("topics_understood", [])
            to_review = entry.get("topics_to_review", [])
            if understood:
                st.markdown("\u2705 " + ", ".join(understood))
            if to_review:
                st.markdown("\u26A0\uFE0F " + ", ".join(to_review))


# ---------------------------------------------------------------------------
# Exercise rendering helper
# ---------------------------------------------------------------------------

def _render_single_exercise(
    index: int,
    exercise,
    exercise_generator,
) -> None:
    """Render a single exercise with answer input and feedback."""
    question = getattr(exercise, "question", str(exercise))
    ex_type = getattr(exercise, "exercise_type", "open_ended")
    options = getattr(exercise, "options", None)
    difficulty = getattr(exercise, "difficulty", "medium")
    topic = getattr(exercise, "topic", "")

    diff_colors = {"easy": "\U0001F7E2", "medium": "\U0001F7E1", "hard": "\U0001F534"}
    diff_icon = diff_colors.get(difficulty, "\u26AA")

    with st.container():
        st.markdown(f"### Question {index + 1} {diff_icon}")
        if topic:
            st.caption(f"Topic: {topic} | Difficulty: {difficulty}")

        st.markdown(question)

        answer_key = f"answer_{index}"
        feedback_key = f"feedback_{index}"

        if ex_type == "multiple_choice" and options:
            answer = st.radio(
                "Select your answer:",
                options,
                key=f"radio_{index}",
                label_visibility="collapsed",
            )
            st.session_state.exercise_answers[answer_key] = answer

        elif ex_type == "true_false":
            answer = st.radio(
                "Select:",
                ["True", "False"],
                key=f"tf_{index}",
                horizontal=True,
                label_visibility="collapsed",
            )
            st.session_state.exercise_answers[answer_key] = answer

        elif ex_type == "code_exercise":
            answer = st.text_area(
                "Write your code:",
                key=f"code_{index}",
                height=150,
                placeholder="Write your solution here...",
            )
            st.session_state.exercise_answers[answer_key] = answer

        else:
            answer = st.text_area(
                "Your answer:",
                key=f"text_{index}",
                height=100,
                placeholder="Type your answer here...",
            )
            st.session_state.exercise_answers[answer_key] = answer

        col_check, col_show = st.columns([1, 1])
        with col_check:
            if st.button("\u2705 Check Answer", key=f"check_{index}"):
                student_answer = st.session_state.exercise_answers.get(answer_key, "")
                if not student_answer:
                    st.warning("Please provide an answer first.")
                else:
                    with st.spinner("Checking..."):
                        try:
                            feedback = exercise_generator.check_answer(
                                exercise, student_answer
                            )
                            st.session_state.exercise_feedback[feedback_key] = feedback

                            prev_key = f"scored_{index}"
                            if prev_key not in st.session_state:
                                st.session_state[prev_key] = True
                                st.session_state.exercise_score["total"] += 1
                                if getattr(feedback, "is_correct", False):
                                    st.session_state.exercise_score["correct"] += 1
                                st.rerun()
                        except Exception as e:
                            st.error(f"Could not check answer: {e}")

        with col_show:
            if st.button("\U0001F4A1 Show Answer", key=f"show_{index}"):
                correct = getattr(exercise, "correct_answer", "N/A")
                explanation = getattr(exercise, "explanation", "")
                source = getattr(exercise, "source_reference", "")
                st.info(f"**Correct answer:** {correct}")
                if explanation:
                    st.markdown(f"**Explanation:** {explanation}")
                if source:
                    st.caption(f"Source: {source}")

        feedback = st.session_state.exercise_feedback.get(feedback_key)
        if feedback:
            is_correct = getattr(feedback, "is_correct", False)
            fb_text = getattr(feedback, "feedback", "")
            explanation = getattr(feedback, "explanation", "")

            if is_correct:
                st.success(f"\u2705 Correct! {fb_text}")
            else:
                st.error(f"\u274C Not quite. {fb_text}")

            if explanation:
                with st.expander("See detailed explanation"):
                    st.markdown(explanation)

        st.markdown("---")
