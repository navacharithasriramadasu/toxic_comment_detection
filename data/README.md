# Dataset

This project uses the **Jigsaw Toxic Comment Classification** dataset from Kaggle.

 https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

## Format


| Column | Description |
|---|---|
| `id` | Unique comment ID |
| `comment_text` | Raw comment text |
| `toxic` | 0 or 1 |
| `severe_toxic` | 0 or 1 |
| `obscene` | 0 or 1 |
| `threat` | 0 or 1 |
| `insult` | 0 or 1 |
| `identity_hate` | 0 or 1 |

## Fine-tuning

```bash
python scripts/train.py --data data/train.csv --output models/bert-toxic-finetuned
```
