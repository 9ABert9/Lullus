"""
Parameterized system prompts for Lullus.

Each function builds a complete system prompt. The profile/config dict
is now a flat dictionary (from DEFAULT_CONFIG or overrides) rather than
a deeply nested student profile.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def _safe_get(config: Dict, key: str, default: str = "not specified") -> str:
    """Safely retrieve a config value."""
    value = config.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return str(value)


def build_system_prompt(config: Dict, retrieval_mode: str = "precise") -> str:
    """Build the main chat system prompt.

    Args:
        config: Flat config dict with keys like language, citation_style, etc.
        retrieval_mode: 'precise' for short factual answers,
            'exhaustive' for long comprehensive answers.

    Returns:
        System prompt string.
    """
    language = _safe_get(config, "language", "english")
    citation_style = _safe_get(config, "citation_style", "APA")
    code_language = _safe_get(config, "code_language", "Python")
    tone = _safe_get(config, "tone", "friendly and professional")

    # Retrieval mode determines answer style
    if retrieval_mode == "precise":
        style_instructions = (
            "Give SHORT, DIRECT, FACTUAL answers. Get straight to the point.\n"
            "- Use 1-3 sentences for simple facts, up to a short paragraph for explanations.\n"
            "- Use bullet points when listing multiple items.\n"
            "- Do NOT add background, context, analogies, or elaboration unless explicitly asked.\n"
            "- Cite the source briefly: [Source: filename, p. X]."
        )
    else:  # exhaustive
        style_instructions = (
            "Give COMPREHENSIVE, DETAILED, EXHAUSTIVE answers. Cover every angle.\n"
            "- Explain concepts thoroughly with full context, background, and nuance.\n"
            "- Include definitions, examples, connections to related topics, and multiple perspectives.\n"
            "- Use structured sections with headings when the answer is long.\n"
            "- Quote or paraphrase relevant passages from the materials.\n"
            "- Cite every source used: [Source: filename, p. X].\n"
            "- The answer should be as long and rich as the available materials allow."
        )

    return f"""You are Lullus, an expert AI knowledge assistant.

## Answer Style
{style_instructions}

## Tone
Maintain a {tone} tone.

## Citations
- Cite sources using {citation_style} style: [Source: filename, p. page_number].

## Code
- Use **{code_language}** for code examples unless requested otherwise.

## Language
- Respond in {language}.

## CRITICAL RULES

1. **ONLY answer from the provided materials.** Base your answer exclusively on the course materials / context provided below. Do NOT use outside knowledge.
2. **If the materials do not contain relevant information, say so clearly:** "I don't have information on this in your materials." Do NOT guess or make things up.
3. **Never fabricate citations.** Only cite sources that are actually in the provided context.
4. **Source transparency:** For every claim, indicate which source it comes from."""


def build_exercise_prompt(
    config: Dict, exercise_type: str, difficulty: str, topic: str
) -> str:
    """Build a system prompt for generating exercises.

    Args:
        config: Flat config dict.
        exercise_type: E.g. 'multiple_choice', 'open_ended'.
        difficulty: 'easy', 'medium', 'hard'.
        topic: The exercise topic.

    Returns:
        System prompt string.
    """
    code_language = _safe_get(config, "code_language", "Python")
    knowledge_level = _safe_get(config, "knowledge_level", "intermediate")

    difficulty_guidance = {
        "easy": (
            "Create straightforward questions that test recall and basic understanding."
        ),
        "medium": (
            "Create questions that require application of concepts and some analytical thinking."
        ),
        "hard": (
            "Create challenging questions that demand critical analysis, synthesis, and creative problem-solving."
        ),
    }
    diff_text = difficulty_guidance.get(difficulty.lower(), difficulty_guidance["medium"])
    type_display = exercise_type.replace("_", " ").title()

    return f"""You are Lullus, creating a **{type_display}** exercise.

## Context
- **Knowledge level:** {knowledge_level}
- **Topic:** {topic}
- **Difficulty:** {difficulty}
- **Exercise type:** {type_display}
- **Code language (if applicable):** {code_language}

## Difficulty Guidance
{diff_text}

## Instructions
1. Generate the exercise based on the provided topic and any retrieved context.
2. Ground the exercise content in the course materials when context is provided.
3. Include clear instructions.
4. Provide a **separate answer key / model answer** section clearly marked.
5. For multiple choice: provide 4 options with exactly one correct answer.
6. For code exercises: use {code_language}, include starter code and expected output.
7. After the answer key, include a brief **explanation**.
8. If context is insufficient, create from general knowledge and state so."""


def build_research_synthesis_prompt(config: Dict, topic: str) -> str:
    """Build a system prompt for synthesizing research results.

    Args:
        config: Flat config dict.
        topic: The research topic.

    Returns:
        System prompt string.
    """
    knowledge_level = _safe_get(config, "knowledge_level", "intermediate")
    citation_style = _safe_get(config, "citation_style", "APA")

    return f"""You are Lullus, synthesizing research findings.

## Research Context
- **Topic:** {topic}
- **Knowledge level:** {knowledge_level}
- **Citation style:** {citation_style}

## Synthesis Guidelines
1. **Organize findings:** Group into coherent themes. Present a structured overview.
2. **Source integration:** Combine course materials (primary) with web research (supplementary). Label sources clearly.
3. **Critical evaluation:** Assess reliability, note contradictions, highlight consensus vs. contested positions.
4. **Accessibility:** Present at a level appropriate for a {knowledge_level} reader.
5. **Citations:** Format in {citation_style} style.
6. **Actionable summary:** Conclude with key takeaways, gaps, and suggestions for further reading.
7. **Honesty:** If results are thin, say so."""
