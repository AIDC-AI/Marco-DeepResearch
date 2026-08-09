from typing import List, Optional, Dict, Any
import re
from umem.extractor.utils import is_prediction_correct

def _construct_memory_block(retrieved_memories: List[Dict[str, Any]]) -> str:
    """Format retrieved memories with stable indices for update operations."""
    if hasattr(retrieved_memories, 'tolist'):
        retrieved_memories = retrieved_memories.tolist()

    if not retrieved_memories or len(retrieved_memories) == 0:
        return ""

    memory_content = ""
    global_idx = 0

    for m in retrieved_memories:
        if 'value' in m and isinstance(m['value'], list):
            for value_str in m['value']:

                memory_content += f"[{global_idx}] {value_str}\n"
                global_idx += 1

    if not memory_content.strip():
        return ""

    return memory_content

def build_extractor_prompt(
    question: str,
    choices: List[str],
    trajectory: str,
    ground_truth_answer: str,
    tokenizer,
    retrieved_memory_item,
):

    choice_txt = ""
    if choices and len(choices) > 0:
        choice_lines = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
        choice_txt = f"\n**Choices**:\n{choice_lines}"

    memory_content = _construct_memory_block(retrieved_memory_item)

    gt_str = str(ground_truth_answer).strip()
    is_correct = False

    if isinstance(trajectory, str):
        score, is_parsed = is_prediction_correct(trajectory, gt_str, choices)
        if score >= 1.0:
            is_correct = True
        think_match = re.search(r"<think>(.*?)</think>", trajectory, flags=re.IGNORECASE | re.DOTALL)
        if think_match:
            trajectory = think_match.group(1).strip()
    else:
        trajectory = ""

    if is_correct:
        status_label = "CORRECT (Success)"
        dynamic_guidelines = """### SCENARIO: SUCCESSFUL EXECUTION
Extract the underlying **Truth** or **Method**.
- **Content**: Abstract the logic. If knowledge, extract the core fact.
- **Constraints**: NO specific options (A/B) or specific numbers/entities.
"""
    else:
        status_label = "INCORRECT (Failure)"
        dynamic_guidelines = """### SCENARIO: FAILED EXECUTION
Analyze the **Root Cause** of the error.
- **Content**: Identify the *type* of confusion or trap.
- **Constraints**: DO NOT simply say "Don't choose X". DO NOT quote the wrong text as a rule.
"""

    system_content = f"""# Role
You are an expert **Experience Summarizer** for a memory bank. Your job is to convert one episode into a reusable, general experience.

# Input Data
1. **User Query**: The problem context.
2. **Past Experiences**: Existing rules (indexed as [0], [1]...).
3. **Model Execution**: The reasoning process.
4. **Execution Status**: Success or Failure.

# CRITICAL CONSTRAINTS
1. **NO ANSWER LEAKAGE**: Never mention specific option indices or answer strings.
2. **NO SPECIFICS**: Remove specific numbers/names. Replace with variables/concepts.
3. **NO HALLUCINATION**: Do not invent facts.

# ACTION GUIDELINES
{dynamic_guidelines}

# MEMORY MANAGEMENT
Compare the new insight with **[Past Experiences]**. Briefly determine whether to **ADD** a new rule or **UPDATE <index>** (replace an existing one).

# Output Format
Strictly follow this structure. Replace the placeholders with the actual reusable experience and operation; do not copy placeholder text.

## Analysis
(Analyze the execution logic. Briefly state if you are ADD or UPDATE <index> based on the comparison.)

## Experience
<experience>
    <value>REUSABLE_EXPERIENCE_TEXT</value>
    <operation>ADD or UPDATE <index></operation>
</experience>
"""

    user_content = f"""# Task Context
[User Query]
{question}
{choice_txt}

[Past Experiences]
{memory_content}

[Model Execution]
{trajectory}

[Execution Status]
**{status_label}**

# Instruction
First, analyze the execution and compare with Past Experiences.
Then, generate the XML block with the experience value and the operation (ADD or UPDATE <index>).
"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt