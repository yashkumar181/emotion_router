from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import logging

# Set up logging for server visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Metadata & Initialization ---
app = FastAPI(
    title="🌿 ArvyaX Wellness Routing API",
    description="A local, robust API that predicts emotional states from noisy text/metadata and routes users to actionable wellness steps.",
    version="1.0.0",
    contact={
        "name": "Yash Kumar"
    }
)

# Add CORS middleware (Crucial for allowing web apps to connect to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
try:
    model_state = joblib.load('state_model.pkl')
    model_intensity = joblib.load('intensity_model.pkl')
    logger.info("ML Models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models. Ensure you run pipeline.py first! Error: {e}")

# --- Input Validation (Pydantic) ---
class UserSession(BaseModel):
    journal_text: str = Field(..., description="The user's raw reflection text.", min_length=2)
    ambience_type: str = Field(default="forest", description="The ambient sound used.")
    time_of_day: str = Field(default="afternoon", description="Time of the session.")
    sleep_hours: float = Field(default=7.0, ge=0.0, le=24.0, description="Hours of sleep last night.")
    energy_level: int = Field(default=3, ge=1, le=5, description="Self-reported energy (1-5).")
    stress_level: int = Field(default=3, ge=1, le=5, description="Self-reported stress (1-5).")

    # This auto-fills the interactive Swagger UI with a realistic test case!
    class Config:
        json_schema_extra = {
            "example": {
                "journal_text": "I feel a bit overwhelmed with all the deadlines today, but the forest sounds helped me breathe.",
                "ambience_type": "forest",
                "time_of_day": "afternoon",
                "sleep_hours": 5.5,
                "energy_level": 2,
                "stress_level": 4
            }
        }

# --- Business Logic ---
def decision_engine(state, intensity, stress, energy, time_of_day, uncertain_flag):
    if uncertain_flag == 1:
        return "pause", "now"
    if state == 'overwhelmed' or stress >= 4:
        if intensity >= 4: return "box_breathing", "now"
        else: return "grounding", "within_15_min"
    if state == 'restless':
        if energy > 3: return "movement", "now"
        else: return "sound_therapy", "tonight"
    if state in ['calm', 'focused']:
        if time_of_day in ['morning', 'afternoon']: return "deep_work", "now"
        else: return "light_planning", "tomorrow_morning"
    if time_of_day == 'night': return "rest", "now"
    else: return "journaling", "later_today"

# --- API Endpoints ---
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint to verify server status."""
    return {"status": "online", "message": "ArvyaX API is running. Navigate to /docs for the Interactive UI."}

@app.post("/api/v1/route", tags=["Routing Engine"])
async def route_user(session: UserSession):
    """
    Ingests user session data, predicts emotional state/intensity, 
    and securely routes the user to the optimal wellness action.
    """
    try:
        # 1. Format input for the ML pipeline
        input_df = pd.DataFrame([{
            'journal_text': session.journal_text,
            'ambience_type': session.ambience_type,
            'time_of_day': session.time_of_day,
            'previous_day_mood': 'missing',  # Gracefully handle missing data
            'face_emotion_hint': 'missing',
            'duration_min': 15,
            'sleep_hours': session.sleep_hours,
            'energy_level': session.energy_level,
            'stress_level': session.stress_level
        }])
        
        # 2. Run Predictions
        pred_state = model_state.predict(input_df)[0]
        confidence = round(float(max(model_state.predict_proba(input_df)[0])), 3)
        
        pred_intensity_raw = model_intensity.predict(input_df)[0]
        pred_intensity = int(np.clip(np.round(pred_intensity_raw), 1, 5))
        
        # 3. Assess Uncertainty
        uncertain_flag = 1 if confidence < 0.45 else 0
        
        # 4. Execute Decision Engine
        what, when = decision_engine(
            pred_state, 
            pred_intensity, 
            session.stress_level, 
            session.energy_level, 
            session.time_of_day, 
            uncertain_flag
        )
        
        # 5. Return structured JSON payload
        return {
            "success": True,
            "understanding": {
                "predicted_state": pred_state,
                "predicted_intensity": pred_intensity,
                "confidence_score": confidence,
                "is_uncertain": bool(uncertain_flag)
            },
            "guidance": {
                "recommended_action": what,
                "timing": when
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Internal ML Pipeline Error.")