"""
Templates for different exercise types in Lullus.

Each template is a string with placeholders for topic, difficulty,
num_questions, context (from RAG), and language. The get_template function
provides a clean lookup interface.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


MULTIPLE_CHOICE_TEMPLATE: str = """Generate {num_questions} multiple-choice question(s) on the topic of **{topic}**.

**Difficulty:** {difficulty}
**Language:** {language}

### Course Context
{context}

### Requirements
- Each question must have exactly **4 options** labeled A, B, C, D.
- Exactly **one option** must be correct.
- Distractors should be plausible and target common misconceptions.
- Questions should test understanding, not just recall, when difficulty is medium or hard.
- Order the options so the correct answer is not always in the same position.

### Output Format
For each question, use this structure:

**Question N:**
[Question text]

A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]

---

### Answer Key
After all questions, provide:

**Question N:** [Correct letter]
- **Explanation:** [Why the correct answer is right and why each distractor is wrong.]
"""


OPEN_ENDED_TEMPLATE: str = """Generate {num_questions} open-ended question(s) on the topic of **{topic}**.

**Difficulty:** {difficulty}
**Language:** {language}

### Course Context
{context}

### Requirements
- Questions should encourage critical thinking, analysis, or synthesis.
- At easy difficulty: ask for explanations or descriptions.
- At medium difficulty: ask for comparisons, applications, or cause-effect analysis.
- At hard difficulty: ask for evaluation, design, or synthesis across multiple concepts.
- Each question should be answerable in 1-3 paragraphs.

### Output Format
For each question:

**Question N:**
[Question text]

---

### Model Answers
After all questions, provide:

**Question N -- Model Answer:**
[A thorough model answer demonstrating the expected depth and quality.]

**Key Points to Cover:**
- [Point 1]
- [Point 2]
- [Point 3]
"""


FILL_IN_BLANK_TEMPLATE: str = """Generate {num_questions} fill-in-the-blank question(s) on the topic of **{topic}**.

**Difficulty:** {difficulty}
**Language:** {language}

### Course Context
{context}

### Requirements
- Each statement should contain exactly **one or two blanks** marked with `________`.
- The blanks should test key terminology, concepts, or relationships.
- Provide enough surrounding context so the answer is unambiguous.
- At easy difficulty: test vocabulary and definitions.
- At medium difficulty: test relationships and applications.
- At hard difficulty: test nuanced distinctions or multi-step reasoning.

### Output Format
For each question:

**Question N:**
[Statement with ________ for each blank.]

---

### Answer Key
After all questions, provide:

**Question N:** [Correct word(s) for each blank]
- **Explanation:** [Brief explanation of why this is the correct answer and its significance.]
"""


TRUE_FALSE_TEMPLATE: str = """Generate {num_questions} true/false question(s) on the topic of **{topic}**.

**Difficulty:** {difficulty}
**Language:** {language}

### Course Context
{context}

### Requirements
- Each statement should be clearly either true or false with no ambiguity.
- Aim for a roughly even split between true and false statements.
- False statements should contain plausible but incorrect claims (common misconceptions).
- At easy difficulty: test basic factual knowledge.
- At medium difficulty: test understanding of relationships or subtle distinctions.
- At hard difficulty: include statements that require careful analysis or are tricky edge cases.

### Output Format
For each question:

**Statement N:**
[Declarative statement]

[ ] True
[ ] False

---

### Answer Key
After all questions, provide:

**Statement N:** [True / False]
- **Explanation:** [Why the statement is true or false, and what the correct fact is if false.]
"""


PROBLEM_SOLVING_TEMPLATE: str = """Generate {num_questions} problem-solving exercise(s) on the topic of **{topic}**.

**Difficulty:** {difficulty}
**Language:** {language}

### Course Context
{context}

### Requirements
- Present a realistic scenario or problem that requires applying course concepts.
- Break complex problems into clearly defined sub-tasks.
- At easy difficulty: single-step or two-step problems with clear data.
- At medium difficulty: multi-step problems requiring integration of several concepts.
- At hard difficulty: open-ended problems with ambiguity, requiring assumptions and justification.
- Include all necessary data, formulas, or reference information the student needs.

### Output Format
For each problem:

**Problem N:**
[Problem description and scenario]

**Given:**
- [Data point 1]
- [Data point 2]

**Tasks:**
1. [Sub-task 1]
2. [Sub-task 2]
3. [Sub-task 3]

---

### Worked Solutions
After all problems, provide:

**Problem N -- Solution:**

**Step 1:** [Description]
[Detailed working]

**Step 2:** [Description]
[Detailed working]

**Final Answer:** [Result with units if applicable]

**Key Concepts Applied:** [List of concepts the student should recognize.]
"""


CODE_EXERCISE_TEMPLATE: str = """Generate {num_questions} coding exercise(s) on the topic of **{topic}**.

**Difficulty:** {difficulty}
**Programming Language:** {language}

### Course Context
{context}

### Requirements
- Each exercise should have a clear problem statement.
- Provide **starter code** with comments indicating where the student should write their solution.
- Include **example input/output** so the student can verify their solution.
- At easy difficulty: straightforward implementation of a single concept.
- At medium difficulty: requires combining multiple concepts or handling edge cases.
- At hard difficulty: algorithmic thinking, optimization, or design patterns required.
- Code must follow best practices for {language} (naming conventions, style, documentation).

### Output Format
For each exercise:

**Exercise N: [Title]**

**Description:**
[Clear problem statement]

**Example:**
```
Input: [example input]
Output: [expected output]
```

**Starter Code:**
```{language}
[Starter code with TODO comments]
```

**Constraints:**
- [Constraint 1]
- [Constraint 2]

---

### Solutions
After all exercises, provide:

**Exercise N -- Solution:**
```{language}
[Complete, working solution with comments]
```

**Explanation:**
[Step-by-step explanation of the approach and key decisions.]

**Time Complexity:** [Big-O analysis]
**Space Complexity:** [Big-O analysis]
"""


_TEMPLATE_REGISTRY: Dict[str, str] = {
    "multiple_choice": MULTIPLE_CHOICE_TEMPLATE,
    "open_ended": OPEN_ENDED_TEMPLATE,
    "fill_in_blank": FILL_IN_BLANK_TEMPLATE,
    "true_false": TRUE_FALSE_TEMPLATE,
    "problem_solving": PROBLEM_SOLVING_TEMPLATE,
    "code_exercise": CODE_EXERCISE_TEMPLATE,
}


def get_template(exercise_type: str) -> str:
    """Return the exercise template for the given type.

    Args:
        exercise_type: One of 'multiple_choice', 'open_ended', 'fill_in_blank',
            'true_false', 'problem_solving', 'code_exercise'.

    Returns:
        The template string with format placeholders.

    Raises:
        ValueError: If exercise_type is not recognized.
    """
    template = _TEMPLATE_REGISTRY.get(exercise_type)
    if template is None:
        available = ", ".join(sorted(_TEMPLATE_REGISTRY.keys()))
        logger.error(
            "Unknown exercise type '%s'. Available types: %s",
            exercise_type,
            available,
        )
        raise ValueError(
            f"Unknown exercise type '{exercise_type}'. "
            f"Available types: {available}"
        )
    logger.debug("Retrieved template for exercise type: %s", exercise_type)
    return template
