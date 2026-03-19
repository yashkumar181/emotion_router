import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    if 'sleep_hours' in df.columns:
        df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].median())
    if 'previous_day_mood' in df.columns:
        df['previous_day_mood'] = df['previous_day_mood'].fillna('missing')
    if 'face_emotion_hint' in df.columns:
        df['face_emotion_hint'] = df['face_emotion_hint'].fillna('missing')
    return df

def decision_engine(state, intensity, stress, energy, time_of_day, uncertain_flag):
    if uncertain_flag == 1:
        return "pause", "now"
        
    if state == 'overwhelmed' or stress >= 4:
        if intensity >= 4:
            return "box_breathing", "now"
        else:
            return "grounding", "within_15_min"
            
    if state == 'restless':
        if energy > 3:
            return "movement", "now"
        else:
            return "sound_therapy", "tonight"
            
    if state in ['calm', 'focused']:
        if time_of_day in ['morning', 'afternoon']:
            return "deep_work", "now"
        else:
            return "light_planning", "tomorrow_morning"
            
    if time_of_day == 'night':
        return "rest", "now"
    else:
        return "journaling", "later_today"

def main():
    train_df = load_and_clean_data("data/train.csv")
    test_df = load_and_clean_data("data/test.csv")

    text_feature = 'journal_text'
    categorical_features = ['ambience_type', 'time_of_day', 'previous_day_mood', 'face_emotion_hint']
    numeric_features = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']

    X_train = train_df[[text_feature] + categorical_features + numeric_features]
    y_state = train_df['emotional_state']
    y_intensity = train_df['intensity']
    
    X_test = test_df[[text_feature] + categorical_features + numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=500, stop_words='english'), text_feature),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ])

    model_state = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model_intensity = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model_state.fit(X_train, y_state)
    model_intensity.fit(X_train, y_intensity)

    joblib.dump(model_state, 'state_model.pkl')
    joblib.dump(model_intensity, 'intensity_model.pkl')
    
    print("Models saved as .pkl files!")

    pred_states = model_state.predict(X_test)
    probas = model_state.predict_proba(X_test)
    confidences = np.max(probas, axis=1)

    pred_intensities_raw = model_intensity.predict(X_test)
    pred_intensities = np.clip(np.round(pred_intensities_raw), 1, 5).astype(int)

    output = pd.DataFrame({'id': test_df['id']})
    output['predicted_state'] = pred_states
    output['predicted_intensity'] = pred_intensities
    output['confidence'] = np.round(confidences, 2)
    output['uncertain_flag'] = (output['confidence'] < 0.45).astype(int)

    whats, whens = [], []
    for i in range(len(test_df)):
        what, when = decision_engine(
            state=pred_states[i],
            intensity=pred_intensities[i],
            stress=test_df.iloc[i]['stress_level'],
            energy=test_df.iloc[i]['energy_level'],
            time_of_day=test_df.iloc[i]['time_of_day'],
            uncertain_flag=output.iloc[i]['uncertain_flag']
        )
        whats.append(what)
        whens.append(when)

    output['what_to_do'] = whats
    output['when_to_do'] = whens

    output.to_csv("predictions.csv", index=False)
    print("Success! predictions.csv has been generated.")

if __name__ == "__main__":
    main()