
# 🛡️ Toxic Comment Detection System

**Summer Internship Project** — Summer of AI (Swecha × IIIT Hyderabad)  
Presented by: R. Shiva & S. Navacharitha

---

## Overview

An AI-driven web application that analyzes user-generated text in **real time** and identifies toxic or abusive comments. The system provides automated moderation actions — **Block**, **Flag**, **Warn**, or **Allow** — based on the predicted toxicity score.

---

## Technology Stack

| Layer | Tools |
|---|---|
| **ML Model** | BERT (`unitary/toxic-bert`) via HuggingFace Transformers |
| **Deep Learning** | PyTorch |
| **NLP Preprocessing** | NLTK, Pandas, NumPy |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | HTML5, CSS3, JavaScript |
| **IDE** | VS Code |

---

## Project Structure

```
toxic-comment-detection/
├── backend/
│   ├── main.py          # FastAPI application & endpoints
│   ├── model.py         # BERT classifier + rule-based fallback
│   └── preprocessor.py  # NLP text preprocessing (NLTK)
├── frontend/
│   └── index.html       # Full UI (HTML/CSS/JS)
├── scripts/
│   ├── setup.py         # Automated setup script
│   ├── train.py         # Fine-tuning script (Jigsaw dataset)
│   ├── create_sample_dataset.py  # Sample dataset generator
│   ├── generate_negative_comments.py  # Test negative comments
│   └── download_dataset.py  # Dataset download helper
├── tests/
│   └── test_api.py      # Pytest test suite
├── data/                # Place Jigsaw CSV dataset here
├── models/              # Fine-tuned model output
└── requirements.txt
```

---

## Quick Start

### Automated Setup (Recommended)

```bash
# Run setup script
python scripts/setup.py

# Manual activation (if needed)
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# Run the server
python backend/main.py

# Open browser
http://localhost:8000
```

### Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create sample dataset
python scripts/create_sample_dataset.py

# 4. Run the server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 5. Open browser
http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Analyze a comment for toxicity |
| `GET` | `/api/health` | Health check + model info |
| `GET` | `/api/stats` | Session statistics |

### Example Request

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "You are so stupid!"}'
```

### Example Response

```json
{
  "text": "You are so stupid!",
  "toxicity_score": 0.8243,
  "is_toxic": true,
  "label": "TOXIC",
  "confidence": 0.8243,
  "categories": {
    "toxic": 0.8243,
    "severe_toxic": 0.1120,
    "obscene": 0.2341,
    "threat": 0.0234,
    "insult": 0.7821,
    "identity_hate": 0.0432
  },
  "action": "FLAG",
  "action_reason": "Content flagged for moderator review"
}
```

---

## Moderation Rules

| Toxicity Score | Action | Description |
|---|---|---|
| ≥ 0.85 | 🚫 **BLOCK** | Severely violates guidelines |
| ≥ 0.60 | 🚩 **FLAG** | Sent to moderator review |
| ≥ 0.35 | ⚠️ **WARN** | User prompted to edit |
| < 0.35 | ✅ **ALLOW** | Meets community standards |

---

## Model Details

The system uses **`unitary/toxic-bert`** from HuggingFace — a BERT model fine-tuned on the Jigsaw Toxic Comment Classification dataset.

**Toxicity categories detected:**
- General Toxicity
- Severe Toxicity
- Obscene Content
- Threatening Language
- Insulting Content
- Identity-Based Hate

### Fine-tuning on Custom Data

```bash
# Download Jigsaw dataset from Kaggle → place in data/train.csv
python scripts/train.py \
  --data data/train.csv \
  --output models/bert-toxic-finetuned \
  --epochs 3
```

---

## Running Tests

```bash
pytest tests/test_api.py -v
```

---

## System Workflow

```
User Input (Frontend)
       ↓
FastAPI Backend receives comment
       ↓
TextPreprocessor cleans text (NLTK)
       ↓
BERT model evaluates toxicity
       ↓
Moderation Rules apply score thresholds
       ↓
Action + scores returned to UI
       ↓
Frontend displays result to user
```

---

## Applications & Future Enhancements

- **Social Media**: Real-time comment moderation; multi-language support
- **Education**: Safe student discussions; behavior analytics
- **Corporate**: Professional messaging enforcement; admin dashboards
- **Customer Support**: Protecting agents from abusive texts
- **Chat & Gaming**: Real-time filtering; lightweight mobile models

---

*Summer of AI · Swecha × IIIT Hyderabad*
=======

"# toxic_comment_detection" 
"# toxic_comment_detection" 
