import re
from verl.utils.reward_score.math import compute_score, is_equiv

def extract_xml_experience(response: str):
    """Extract the memory key and values from an <experience> XML block."""
    if not response:
        return None, []

    exp_pattern = re.compile(r"<experience>(.*?)</experience>", re.DOTALL | re.IGNORECASE)
    exp_match = exp_pattern.search(response)

    if not exp_match:
        return None, []

    content = exp_match.group(1)

    key_pattern = re.compile(r"<key>(.*?)</key>", re.DOTALL | re.IGNORECASE)
    key_match = key_pattern.search(content)

    if not key_match:
        return None, []

    key_str = key_match.group(1).strip()

    value_pattern = re.compile(r"<value>(.*?)</value>", re.DOTALL | re.IGNORECASE)
    raw_values = value_pattern.findall(content)

    value_list = [v.strip() for v in raw_values if v.strip()]

    return key_str, value_list

def _extract_xml_experience(response: str):
    full_pattern = re.compile(
        r"^\s*<experience>\s*"
        r"<key>(?P<key_content>.*?)</key>"
        r"(?P<value_group>(?:\s*<value>.*?</value>)*)"
        r"\s*</experience>\s*$",
        re.DOTALL | re.IGNORECASE
    )
    match = full_pattern.fullmatch(response)
    if not match:
        return None, []
    key_str = match.group("key_content").strip()
    value_group_content = match.group("value_group")
    value_list = [v.strip() for v in re.findall(r"<value>(.*?)</value>", value_group_content, re.DOTALL | re.IGNORECASE)]
    return key_str, value_list

def extract_think_and_solution(response: str):
    start_tag = "<think>"
    end_tag = "</think>"

    start_idx = response.find(start_tag)
    end_idx = response.find(end_tag)

    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return "", response

    think_str = response[start_idx + len(start_tag) : end_idx].strip()

    solution_str = response[end_idx + len(end_tag) :].strip()

    return think_str, solution_str

def _is_prediction_correct(trajectory, ground_truth, choices=None):
    """Return whether a trajectory answer matches the ground truth."""

    traj = str(trajectory)

    m = re.search(r"<answer>(.*?)</answer>", traj, flags=re.IGNORECASE | re.DOTALL)

    if m:
        pred = m.group(1)
    else:

        return False

    pred = pred.strip()
    pred = " ".join(pred.split())
    pred_norm = pred.lower()

    gt = str(ground_truth)
    gt = gt.strip()
    gt = " ".join(gt.split())
    gt_norm = gt.lower()

    is_correct = (pred_norm == gt_norm) or bool(compute_score(pred_norm, gt_norm))

    return float(is_correct)

def is_prediction_correct(trajectory, ground_truth, choices=None):
    """Score a trajectory answer and report whether an answer tag was parsed."""

    traj = str(trajectory)

    m = re.search(r"<answer>(.*?)</answer>", traj, flags=re.IGNORECASE | re.DOTALL)

    parsed = False

    if m:
        pred = m.group(1)
        parsed = True

    if not parsed:
        return 0.0, False

    def safe_float_compare(a, b):
        try:

            return float(a) == float(b)
        except ValueError:
            try:

                from fractions import Fraction
                return float(Fraction(a)) == float(Fraction(b))
            except Exception:

                return False

    is_correct = safe_float_compare(pred, ground_truth) or\
                (str(pred).strip() == str(ground_truth).strip()) or\
                bool(is_equiv(pred, ground_truth))

    return (1.0, True) if is_correct else (0.0, True)