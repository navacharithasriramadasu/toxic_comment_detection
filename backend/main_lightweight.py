"""
Lightweight Toxic Comment Detection System - FastAPI Backend
Uses rule-based classifier for systems with limited memory
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Toxic Comment Detection API",
    description="Lightweight toxic comment detection using rule-based classifier",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class CommentRequest(BaseModel):
    text: str
    user_id: Optional[str] = "anonymous"

class ToxicityResult(BaseModel):
    text: str
    toxicity_score: float
    is_toxic: bool
    label: str
    confidence: float
    categories: dict
    action: str
    action_reason: str
    model_type: str

# --- Load Lightweight Model ---
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a lightweight classifier that always uses rule-based method
class LightweightToxicityClassifier:
    """Lightweight rule-based toxicity classifier for low-memory systems"""
    
    TOXICITY_CATEGORIES = [
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate"
    ]

    CATEGORY_LABELS = {
        "toxic": "General Toxicity",
        "severe_toxic": "Severe Toxicity",
        "obscene": "Obscene Content",
        "threat": "Threatening Language",
        "insult": "Insulting Content",
        "identity_hate": "Identity-Based Hate",
    }

    def __init__(self):
        self.model_name = "rule-based-lightweight"
        self.device = "cpu"
        self._stats = {"total_analyzed": 0, "toxic_detected": 0, "blocked": 0}
        
        # Enhanced toxic patterns for better detection
        self.toxic_patterns = {
            "severe_toxic": [
                "kill yourself", "kys", "die", "murder", "destroy you", "end your life",
                "commit suicide", "take your own life", "harm yourself", "hurt yourself"
            ],
            "threat": [
                "i will hurt", "i'll kill", "you're dead", "watch your back", "coming for you",
                "find you", "hunt you down", "get you", "payback", "revenge"
            ],
            "obscene": [
                "fuck", "shit", "ass", "bitch", "cunt", "damn", "bastard", "whore", "slut",
                "piss", "crap", "hell", "goddamn"
            ],
            "insult": [
                "idiot", "stupid", "moron", "loser", "dumb", "ugly", "pathetic", "fool",
                "jerk", "asshole", "douchebag", "retard", "imbecile", "nitwit"
            ],
            "identity_hate": [
                "racist", "sexist", "homophobic", "slur", "nazi", "fascist", "bigot",
                "discriminate", "prejudice", "hate group"
            ],
            "toxic": [
                "hate", "terrible", "awful", "worst", "horrible", "disgust", "sucks",
                "garbage", "trash", "useless", "worthless", "pathetic"
            ],
        }

    def predict(self, text: str) -> dict:
        """Run toxicity prediction using rule-based approach"""
        self._stats["total_analyzed"] += 1

        result = self._rule_based_predict(text)

        if result["is_toxic"]:
            self._stats["toxic_detected"] += 1

        return result

    def _rule_based_predict(self, text: str) -> dict:
        """Enhanced rule-based toxicity scoring"""
        text_lower = text.lower()
        categories = {}
        max_score = 0.0

        for category, keywords in self.toxic_patterns.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            # Enhanced scoring with weight based on severity
            if category == "severe_toxic":
                score = min(matches * 0.5, 1.0)  # Higher weight for severe toxicity
            elif category == "threat":
                score = min(matches * 0.45, 1.0)
            elif category == "obscene":
                score = min(matches * 0.3, 1.0)
            elif category == "insult":
                score = min(matches * 0.25, 1.0)
            elif category == "identity_hate":
                score = min(matches * 0.4, 1.0)
            else:
                score = min(matches * 0.2, 1.0)
            
            categories[category] = round(score, 4)
            if score > max_score:
                max_score = score

        toxicity_score = min(max_score, 1.0)
        
        # Boost if multiple categories triggered
        triggered = sum(1 for v in categories.values() if v > 0)
        if triggered >= 2:
            toxicity_score = min(toxicity_score + 0.15, 1.0)
        elif triggered >= 3:
            toxicity_score = min(toxicity_score + 0.25, 1.0)

        # Adjusted threshold for better sensitivity
        is_toxic = toxicity_score >= 0.3
        confidence = toxicity_score if is_toxic else (1 - toxicity_score)

        return {
            "toxicity_score": round(toxicity_score, 4),
            "is_toxic": is_toxic,
            "label": "TOXIC" if is_toxic else "NON-TOXIC",
            "confidence": round(confidence, 4),
            "categories": categories,
        }

    def get_stats(self) -> dict:
        total = self._stats["total_analyzed"]
        toxic = self._stats["toxic_detected"]
        return {
            "total_analyzed": total,
            "toxic_detected": toxic,
            "non_toxic": total - toxic,
            "toxic_rate": round(toxic / total * 100, 2) if total > 0 else 0,
            "model": self.model_name,
            "mode": "rule-based-lightweight",
        }

# Initialize lightweight classifier
classifier = LightweightToxicityClassifier()

# Simple text preprocessor
class SimplePreprocessor:
    def clean(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        return text.strip()

preprocessor = SimplePreprocessor()

# --- Moderation Rules ---
def apply_moderation_rules(score: float, categories: dict) -> tuple[str, str]:
    """Rule-based moderation logic"""
    severe_toxic = categories.get("severe_toxic", 0)
    threat = categories.get("threat", 0)
    obscene = categories.get("obscene", 0)

    if score >= 0.8 or severe_toxic >= 0.5 or threat >= 0.4:
        return "BLOCK", "Content severely violates community guidelines"
    elif score >= 0.5 or obscene >= 0.6:
        return "FLAG", "Content flagged for moderator review"
    elif score >= 0.3:
        return "WARN", "Please review and edit your comment before posting"
    else:
        return "ALLOW", "Comment meets community standards"

# --- Endpoints ---
@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "../frontend/index.html"))

@app.post("/api/analyze", response_model=ToxicityResult)
async def analyze_comment(request: CommentRequest):
    """Analyze a comment for toxicity using lightweight rule-based model"""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Comment text cannot be empty")

    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Comment exceeds maximum length of 5000 characters")

    try:
        # Preprocess text
        cleaned_text = preprocessor.clean(request.text)

        # Get model prediction
        result = classifier.predict(cleaned_text)

        # Apply moderation rules
        action, reason = apply_moderation_rules(result["toxicity_score"], result["categories"])

        logger.info(f"Analyzed comment | Score: {result['toxicity_score']:.3f} | Action: {action}")

        return ToxicityResult(
            text=request.text,
            toxicity_score=round(result["toxicity_score"], 4),
            is_toxic=result["is_toxic"],
            label=result["label"],
            confidence=round(result["confidence"], 4),
            categories=result["categories"],
            action=action,
            action_reason=reason,
            model_type="rule-based-lightweight"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model": classifier.model_name,
        "device": classifier.device,
        "mode": "rule-based-lightweight",
        "memory_usage": "low"
    }

@app.get("/api/stats")
def get_stats():
    return classifier.get_stats()

if __name__ == "__main__":
    logger.info("Starting lightweight toxic comment detection server...")
    logger.info("Model: Rule-based (low memory usage)")
    uvicorn.run("main_lightweight:app", host="0.0.0.0", port=8000, reload=True)
