# Decision Engine: Predict, Understand, Guide

This is a local, end-to-end Python pipeline designed to understand messy user data, predict emotional states and recommend actionable wellness steps.

## Approach & Architecture

Rather than relying on a heavy deep-learning model or a black-box LLM API, I designed a **Hybrid Rule-Augmented ML Pipeline**. 
1. **Machine Learning Layer (Predict):** A `RandomForestClassifier` combined with `TfidfVectorizer` to handle noisy text, alongside `OneHotEncoder` and `StandardScaler` for the contextual metadata.
2. **Business Logic Layer (Decide & Guide):** A  routing engine (`decision_engine`) that takes the ML predictions and routes the user safely to the correct wellness action.

## Feature Importance & Ablation Study
During the EDA phase, I conducted an study to compare a Text-Only model against the Hybrid (Text + Metadata) model.
* **Text-Only Model:** Reached an accuracy ceiling of ~45%. It struggled heavily with short, vague inputs (e.g., "fine") because it lacked context.
* **Text + Metadata Model:** Reached ~63% accuracy. 
* **Feature Understanding:** Metadata features like `stress_level` and `sleep_hours` were the most critical deciding factors when the `journal_text` was contradictory. The metadata acted as the "ground truth" for the user's physiological state, overriding sarcastic or vague text tokens.

### Feature Engineering (Handling the Messiness)
* **Text (`journal_text`):** Handled via `TfidfVectorizer` to ensure fast, local execution without needing heavy tokenizers. Stop words were removed to focus on core sentiment tokens.
* **Missing Metadata:** * Missing numeric values (`sleep_hours`) were median-imputed.
  * Missing categorical values (`face_emotion_hint`) were imputed with the string `"missing"`. This allows the model to learn if the *absence* of a facial cue is itself a predictive feature.

### Model Choice
I selected **Random Forest** for both classification (emotional state) and regression (intensity).
* **Why not Neural Networks?** RFs are natively robust to outliers, handle mixed tabular/text data beautifully, and don't require heavy GPU tuning. 
* **Intensity as Regression:** I chose `RandomForestRegressor` for intensity (1-5) because predicting a 3 when the truth is a 4 is "less wrong" than predicting a 1. Regression naturally penalizes this continuous distance, after which I rounded the prediction to the nearest integer.

## Setup & How to Run

1.  Ensure you have a standard Python environment with `pandas`, `numpy`, and `scikit-learn` installed.
2.  Place your datasets inside a folder named `data/` (`data/train.csv` and `data/test.csv`).
3.  Run the pipeline:
    ```bash
    python pipeline.py
    ```
4.  The output will be saved as `predictions.csv` in the root directory

## Interactive UI Demo
I have built a local web dashboard. 

To run the interactive app locally:
1. Ensure `streamlit` is installed (`pip install streamlit`).
2. Run this command in your terminal:
   ```bash
   streamlit run app.py