# Edge Deployment Strategy

To align with ArvyaX's mission of creating accessible intelligence, my model must eventually run locally on a user's device (mobile/offline) to preserve privacy and reduce API costs.

## Deployment Approach
The current `RandomForest` + `TF-IDF` pipeline is already lightweight. However, to deploy this natively to iOS/Android, we cannot ship standard Python binaries. 

1. **Model Conversion (ONNX):** We would convert the trained scikit-learn Pipeline into the **ONNX (Open Neural Network Exchange)** format using `skl2onnx`. 
2. **Runtime:** The mobile app would use the ONNX Runtime to execute inferences directly on the device.

## Optimizations & Trade-offs
* **Model Size:** The current TF-IDF vocabulary is capped at `max_features=500`. Combined with a Random Forest `n_estimators=100`, the total ONNX file size would be roughly **1MB to 3MB**. This is perfectly acceptable for mobile app bundling.
* **Latency:** Local inference on an iPhone/Android via ONNX takes less than `10ms`, allowing for real-time routing immediately after the user finishes typing.
* **The Trade-off:** By restricting the vocabulary size to 500 features to keep the model small, we lose edge-case vocabulary comprehension. If a user inputs highly complex vocabulary, it will be treated as an unknown token, likely triggering the `uncertain_flag`.