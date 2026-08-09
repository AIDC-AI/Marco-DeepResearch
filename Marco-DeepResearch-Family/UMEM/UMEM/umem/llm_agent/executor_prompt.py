"""Prompt builders for the executor LLM."""

from typing import Dict, Any, List, Optional

def _construct_memory_block(retrieved_memories: List[Dict[str, Any]]) -> str:
    """Format retrieved memory values as a prompt section."""
    if hasattr(retrieved_memories, 'tolist'):
        retrieved_memories = retrieved_memories.tolist()

    if not retrieved_memories or len(retrieved_memories) == 0:
        return ""

    memory_content = ""

    for m in retrieved_memories:

        if 'value' in m and isinstance(m['value'], list):

            for value_str in m['value']:

                memory_content += f"  - {value_str}\n"

    if not memory_content.strip():
        return ""

    return memory_content

def build_executor_cq_prompt(
    question: str,
    choices: List[str],
    retrieved_memories: List[Dict[str, Any]],
) -> str:
    """Build a multiple-choice executor prompt."""
    memory_section = _construct_memory_block(retrieved_memories)

    choice_block = "\n".join(
        [f"{i}. {text}" for i, text in enumerate(choices)]
    )

    prompt = f"""
# Role
You are an expert Task Execution Agent. Your goal is to solve multiple-choice questions by applying **Past Effective Experiences**.

# Input Data
1. **Past Experiences**: Historical context or rules to guide your decision.
2. **Question**: The specific problem you need to solve.
3. **Options**: A list of candidate answers.

# Instructions
1. Analyze the **Question** carefully.
2. Refer to the **Past Experiences** to find the logic or evidence required to solve the problem.
3. Evaluate the **Options** and select the best one.
4. **CRITICAL**: Identify the **Index** of the selected option based on a **0-based system** (i.e., the first item is 0, the second is 1, etc.).

# Output Format
You must strictly follow this format:
1. First, analyze the problem and the options (reasoning).
2. Then, append the answer tag.

Structure:
...reasoning process... <answer>Index</answer>

---
# Current Task
**Past Experiences**:
{memory_section}

**Question**:
{question}

**Options**:
{choice_block}

**Output**:
"""

    return prompt.strip()

def build_executor_math_prompt(
    question: str,
    retrieved_memories: List[Dict[str, Any]],
) -> str:
    """Build a free-form math executor prompt."""
    memory_section = _construct_memory_block(retrieved_memories)

    prompt = f"""
    # Role
    You are an expert Math Task Execution Agent. Your goal is to solve mathematical problems by applying logic and methods from **Past Effective Experiences**.

    # Input Data
    1. **Past Experiences**: Relevant formulas, theorems, or similar solved examples.
    2. **Question**: The specific math problem you need to solve.

    # Instructions
    1. Analyze the **Question** to identify the mathematical concepts involved.
    2. Refer to the **Past Experiences** to find the correct formula, method, or logic pattern.
    3. Perform the **Problem Solving Process** step-by-step. Show your work, calculations, and derivations clearly.

    # Output Format
    You must strictly follow this format:
    1. First, provide the **Problem Solving Process** (natural language mixed with LaTeX equations if needed).
    2. Finally, append the answer tag with the final result.

    Structure:
    ...problem solving process... <answer>Final_Result</answer>

    # Current Task
    **Past Experiences**:
    {memory_section}

    **Question**:
    {question}

    **Output**:

    """

    return prompt.strip()

def build_executor_prompt(
    question: str,
    choices: Optional[List[str]],
    retrieved_memories: List[Dict[str, Any]]
) -> str:
    """
    Construct a SINGLE prompt, automatically selecting between MCQ and Math prompt styles.
    """

    if choices and isinstance(choices, list) and len(choices) > 0:
        return build_executor_cq_prompt(question, choices, retrieved_memories)
    else:

        return build_executor_math_prompt(question, retrieved_memories)