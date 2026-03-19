import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="ArvyaX Wellness Router")

@st.cache_resource
def load_models():
    state_model = joblib.load('state_model.pkl')
    intensity_model = joblib.load('intensity_model.pkl')
    return state_model, intensity_model

model_state, model_intensity = load_models()

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

def generate_supportive_message(state, what_to_do, uncertain_flag):
    if uncertain_flag == 1:
        return "It sounds like there's a lot going on right now, and that's okay. Let's just take a brief pause to reset."
    
    messages = {
        "overwhelmed": f"I hear you. Things feel heavy right now. Let's slow things down and try some {what_to_do.replace('_', ' ')}.",
        "restless": f"You seem to have some scattered energy. Channeling that into {what_to_do.replace('_', ' ')} might help you find your center.",
        "calm": f"You're in a great headspace! Let's lean into this clarity with some {what_to_do.replace('_', ' ')}.",
        "focused": f"You are dialed in. This is the perfect time for {what_to_do.replace('_', ' ')} to make the most of your energy.",
        "mixed": f"It’s totally normal to feel a mix of things. Let's gently shift our focus to {what_to_do.replace('_', ' ')}.",
        "neutral": f"You're holding steady. A little {what_to_do.replace('_', ' ')} might be exactly what you need to transition through your day."
    }
    return messages.get(state, "Take a deep breath. You've got this.")

st.title("🌿 ArvyaX Wellness Router")
st.markdown("### From Understanding Humans → To Guiding Them")

with st.form("user_input_form"):
    st.subheader("1. Your Reflection")
    journal_text = st.text_area("How are you feeling after your session?", placeholder="Type your reflection here...")
    
    st.subheader("2. Contextual Signals")
    col1, col2 = st.columns(2)
    with col1:
        stress_level = st.slider("Stress Level", 1, 5, 3)
        energy_level = st.slider("Energy Level", 1, 5, 3)
        time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"])
    with col2:
        sleep_hours = st.number_input("Sleep Hours Last Night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        ambience_type = st.selectbox("Ambience Session Used", ["forest", "ocean", "rain", "mountain", "cafe"])
        
    submit_button = st.form_submit_button(label="Analyze & Guide")

if submit_button and journal_text:
    input_df = pd.DataFrame({
        'journal_text': [journal_text],
        'ambience_type': [ambience_type],
        'time_of_day': [time_of_day],
        'previous_day_mood': ['missing'],
        'face_emotion_hint': ['missing'],
        'duration_min': [15],
        'sleep_hours': [sleep_hours],
        'energy_level': [energy_level],
        'stress_level': [stress_level]
    })
    
    pred_state = model_state.predict(input_df)[0]
    probas = model_state.predict_proba(input_df)[0]
    confidence = round(max(probas), 2)
    
    pred_intensity_raw = model_intensity.predict(input_df)[0]
    pred_intensity = int(np.clip(np.round(pred_intensity_raw), 1, 5))
    
    uncertain_flag = 1 if confidence < 0.45 else 0
    
    what, when = decision_engine(pred_state, pred_intensity, stress_level, energy_level, time_of_day, uncertain_flag)
    
    supportive_msg = generate_supportive_message(pred_state, what, uncertain_flag)
    
    st.divider()
    st.subheader(" System Diagnostics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted State", pred_state.capitalize())
    c2.metric("Intensity (1-5)", pred_intensity)
    c3.metric("Confidence", f"{confidence*100}%", delta="Uncertain!" if uncertain_flag else "High", delta_color="inverse" if uncertain_flag else "normal")
    
    st.subheader("Recommended Action")
    st.info(f"**Action:** {what.replace('_', ' ').title()} \n\n**When:** {when.replace('_', ' ').title()}")
    
    st.subheader(" ArvyaX Companion Message")
    st.success(f"*{supportive_msg}*")