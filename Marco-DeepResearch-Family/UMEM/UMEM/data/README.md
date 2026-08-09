# Data

This directory contains the default UMEM training and validation data:

- `train.parquet`: 2,281 training rows
- `test.parquet`: 70 validation rows

The default paths are already configured in `.env.example`:

```bash
TRAIN_FILE_PATH=data/train.parquet
TEST_FILE_PATH=data/test.parquet
```

You can replace these files with your own Parquet data by setting `TRAIN_FILE_PATH` and `TEST_FILE_PATH` before running `scripts/train_umem.sh`.

## Schema

Each row is read by `verl.utils.dataset.RLHFDataset` and should contain:

| Field | Description |
| --- | --- |
| `prompt` | Chat-style messages, for example `[{"role": "user", "content": "..."}]`. |
| `data_source` | Dataset or task name. |
| `reward_model` | Object containing at least `ground_truth`; `style` is also supported. |
| `extra_info` | Object containing `question`; multiple-choice tasks should include `choices`. |

## Semantic Neighborhood Fields

Semantic Neighborhood Modeling uses these optional lists under `extra_info`:

| Field | Description |
| --- | --- |
| `similar_queries` | Semantically related questions used to evaluate memory generalization. |
| `similar_choices` | Choices for each related question; use empty lists for free-form tasks. |
| `similar_gt` | Ground-truth answers for the related questions. |

The three `similar_*` lists should have the same length. If they are omitted or empty, UMEM evaluates memory utility on the current query.

## Example

```json
{
  "prompt": [{"role": "user", "content": "Question text"}],
  "data_source": "mmlu",
  "reward_model": {"style": "rule", "ground_truth": "2"},
  "extra_info": {
    "question": "Question text",
    "choices": ["A", "B", "C", "D"],
    "similar_queries": ["Semantically related question text"],
    "similar_choices": [["A", "B", "C", "D"]],
    "similar_gt": ["2"]
  }
}
```
