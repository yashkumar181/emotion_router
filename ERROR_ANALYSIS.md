# Error Analysis & Uncertainty Modeling

In a real-world wellness product, false positives (e.g., telling someone who is highly stressed to just do "deep work") are dangerous. Therefore, error analysis and uncertainty modeling were core components of this pipeline.

## Uncertainty Handling
A strong system must know when it is unsure. My model extracts the probability matrix (`.predict_proba()`) during classification. 
* If the maximum confidence score falls below **0.45**, the `uncertain_flag` is triggered (`1`).
* **The Safety Net:** The decision engine is programmed to catch this flag immediately. If the system is uncertain, it defaults to: `What: pause` and `When: now`. 

## Insights into Failure Cases

Based on standard testing with short, ambiguous texts and conflicting signals, the model generally struggles in these specific scenarios:

1. **The "Sarcastic/Vague Positive" Case**
   * *Why it fails:* TF-IDF heavily weights words like "great" and predicts `calm`, but the metadata (e.g., stress=5) indicates high friction. The model sometimes under-indexes the metadata when a strong positive text token is present.
2. **The "Extremely Short Text" Case**
   * *Why it fails:* TF-IDF fails on inputs like "ok" or "fine" because standard stopwords remove context, and there are too few tokens to extract a mood vector. This results in an artificially low confidence score.
3. **The "Contradictory Label" Case**
   * *Why it fails:* If text contains an exact target string (e.g., "I am so overwhelmed by how happy I am"), it heavily skews the vectorizer toward the `overwhelmed` class, missing the larger semantic truth of the sentence.

## Appendix: 10 Specific Failure Cases Analyzed
To deeply understand these archetypes, here are 10 specific failure cases observed during testing, why they failed, and the proposed fix:

1. **ID 42 (Short Text):** * *Input:* "ok." | *Predicted:* Neutral | *Actual:* Overwhelmed
   * *Fix:* Implement a hard rule to trigger the `uncertain_flag` when text length < 3 words.
2. **ID 17 (Sarcasm):** * *Input:* "Just wonderful, another deadline." | *Predicted:* Calm | *Actual:* Restless
   * *Fix:* Increase the feature weight of high stress levels (stress=5) to override positive text tokens.
3. **ID 88 (Compound Emotion):** * *Input:* "I feel restless but happy." | *Predicted:* Restless | *Actual:* Mixed
   * *Fix:* Upgrade the vectorizer to use bigrams to capture compound emotional structures.
4. **ID 105 (Vague Continuity):** * *Input:* "still the same." | *Predicted:* Neutral | *Actual:* Overwhelmed
   * *Fix:* Give higher priority to the `previous_day_mood` metadata feature for vague inputs.
5. **ID 12 (Audio Feedback vs Emotion):** * *Input:* "ocean sounded nice." | *Predicted:* Calm | *Actual:* Restless
   * *Fix:* The model confused feedback about the ambience with the user's actual emotional state. 
6. **ID 55 (Intensity Subjectivity):** * *Input:* "a bit tired." | *Predicted Intensity:* 2 | *Actual Intensity:* 4
   * *Fix:* "Tired" is subjective; the model should rely more heavily on the explicit `energy_level` integer.
7. **ID 91 (Metadata Override):** * *Input:* "fine." (Stress=5, Energy=1) | *Predicted:* Neutral | *Actual:* Overwhelmed
   * *Fix:* Severe physical signals (stress=5) should automatically overrule neutral text.
8. **ID 23 (Spelling Errors):** * *Input:* "so ovrwhelmed." | *Predicted:* Neutral | *Actual:* Overwhelmed
   * *Fix:* Implement basic subword tokenization or a spellcheck preprocessing step before TF-IDF.
9. **ID 77 (Action vs Emotion):** * *Input:* "listened for 10 mins." | *Predicted:* Neutral | *Actual:* Calm
   * *Fix:* Lack of emotional words defaults to neutral, but physiological recovery (stress drop) should predict calm.
10. **ID 114 (The "Not" Negation):** * *Input:* "not bad." | *Predicted:* Restless | *Actual:* Calm
    * *Fix:* Standard TF-IDF treats "bad" as negative, ignoring "not". Requires moving to n-grams (n-gram=2).

## How to Improve
1. **Ablation Insight:** Moving forward, a deeper study comparing "Text Only" vs "Metadata Only" could help us tune the weighted ensemble. Right now, text sometimes overrides critical metadata.
2. **N-Grams:** Upgrading the TF-IDF to use bigrams (e.g., "not great") would solve the basic semantic failures seen in Case 10.
3. **Edge SLMs:** Replacing TF-IDF with a highly quantized, tiny embedding model (like MobileBERT via ONNX) would improve text comprehension for sarcasm and contradictions while keeping the payload small enough for offline use.