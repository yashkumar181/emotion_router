# Error Analysis & Uncertainty Modeling

In a real world wellness product, false positives (e.g., telling someone who is highly stressed to just do "deep work") are dangerous. Therefore, error analysis and uncertainty modeling were core components of this pipeline.

## Uncertainty Handling
A strong system must know when it is unsure. My model extracts the probability matrix (`.predict_proba()`) during classification. 
* If the maximum confidence score falls below **0.45**, the `uncertain_flag` is triggered (`1`).
* **The Safety Net:** The decision engine is programmed to catch this flag immediately. If the system is uncertain, it defaults to: `What: pause` and `When: now`. 

## Insights into Failure Cases

Based on standard testing with short, ambiguous texts and conflicting signals, the model struggles in these specific scenarios:

1. **The "Sarcastic/Vague Positive" Case**
   * *Example Input:* "Wow, everything is just great." (With stress level = 5).
   * *Why it fails:* TF-IDF heavily weights the word "great" and predicts `calm`, but the metadata (stress=5) indicates high friction. The model sometimes under-indexes the metadata when a strong positive text token is present.
2. **The "Extremely Short Text" Case**
   * *Example Input:* "ok" or "fine"
   * *Why it fails:* TF-IDF fails here because standard stopwords remove context, and there are too few tokens to extract a mood vector. This results in an artificially low confidence score.
3. **The "Contradictory Label" Case**
   * *Example Input:* "I am so overwhelmed by how happy I am today."
   * *Why it fails:* The text contains the exact string "overwhelmed", which heavily skews the vectorizer toward the `overwhelmed` class, missing the larger semantic truth of the sentence.

## How to Improve
1. **Insight:** Moving forward, a deeper study comparing "Text Only" vs "Metadata Only" could help us tune the weighted ensemble. Right now, text sometimes overrides critical metadata.
2. **N-Grams:** Upgrading the TF-IDF to use bigrams (e.g., "not great") would solve some of the basic semantic failures.
3. **Edge SLMs:** Replacing TF-IDF with a highly quantized, tiny embedding model (like MobileBERT) would improve text comprehension while keeping the payload small enough for offline use.