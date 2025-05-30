# In-Depth Overview: Iterative Contrastive Learning Pipeline

This document outlines the structure and workflow of the refactored Iterative Contrastive Learning Pipeline. The primary goal of this pipeline is to improve class representations by iteratively identifying prediction errors (specifically false positives), incorporating them as negative examples, and retraining the model.

## I. Core Philosophy

The pipeline operates on the principle of iterative refinement. Instead of a single training pass, it goes through multiple cycles:

1.  **Train:** Learn class representatives using contrastive learning.
2.  **Predict:** Use these representatives to make predictions on a dataset.
3.  **Evaluate:** Compare predictions against ground truth to identify errors.
4.  **Adapt:** Use identified false positives to augment the training data for the next iteration, teaching the model what *not* to predict for certain classes.
5.  **Repeat:** Continue this cycle until a convergence criterion is met or a maximum number of iterations is reached.

The current prediction strategy focuses on using only the *positive class representatives* for final similarity scoring, after these representatives have been implicitly refined by training against "not\_[class]" examples.

## II. Directory Structure & Modules

The refactored pipeline logic resides primarily within the `iterative_pipeline` package:

```
contrastive_learning_v2/
├── iterative_pipeline/         # Core pipeline package
│   ├── __init__.py             # Package marker
│   ├── config.py               # Configuration dataclass (PipelineConfig)
│   ├── data_utils.py           # Data loading and validation
│   ├── training_utils.py       # Wrapper for the external training script
│   ├── prediction_utils.py     # Prediction generation logic
│   ├── evaluation_utils.py     # Evaluation and false positive extraction
│   ├── iteration_manager.py    # Logic for a single pipeline iteration
│   └── pipeline_manager.py     # Orchestrates the overall multi-iteration pipeline
│
├── run_iterative_pipeline.py   # Main script to execute the pipeline
│
└── train_representatives.py    # External script for actual contrastive training
```

## III. Key Components and Workflow

### 1. Entry Point: `run_iterative_pipeline.py`

*   **Responsibilities:**
    *   Parses all command-line arguments (e.g., data paths, iteration counts, thresholds, margins).
    *   Instantiates a `PipelineConfig` object from `iterative_pipeline.config` to hold these parameters.
    *   Calls `load_and_validate_data` from `iterative_pipeline.data_utils` to load initial datasets (`definitiveObjects`, `resnetPredictions`, `trackingInfo`).
    *   Invokes the main pipeline execution function `run_iterative_pipeline` from `iterative_pipeline.pipeline_manager`.
    *   Provides top-level error handling and prints tracebacks.

### 2. Configuration: `iterative_pipeline/config.py`

*   **`PipelineConfig` Dataclass:** A structured way to manage and pass all pipeline parameters. This includes:
    *   Paths to input data files.
    *   Iteration control (max iterations, convergence F1 threshold).
    *   Prediction thresholds (primary and an optional secondary for later iterations).
    *   Training parameters for contrastive learning (epochs, primary margin, optional secondary margin).
    *   Output directory and test mode flag.

### 3. Data Handling: `iterative_pipeline/data_utils.py`

*   **`load_and_validate_data()` function:**
    *   Loads the three primary pickle files:
        *   `definitiveObjects.pkl`: Initial positive training examples with `class` and `finetuned_embedding`.
        *   `resnetPredictions.pkl`: A larger dataset of embeddings (likely frame-level) with `video`, `frame`, `owl_label` (true class at frame level), and `finetuned_embedding`.
        *   `trackingInfo.pkl`: Ground truth for video-level presence/absence of classes, with `video`, `tag` (class), and `actual` (0 or 1).
    *   Performs basic validation (e.g., checks for required columns, handles dict-to-DataFrame conversion).
    *   Standardizes `trackingInfo` by grouping to get the max `actual` value per video-tag pair.

### 4. Training Wrapper: `iterative_pipeline/training_utils.py`

*   **`train_contrastive_representatives()` function:**
    *   This function *does not* implement contrastive learning itself. Instead, it acts as a wrapper to call your existing `train_representatives.py` script.
    *   **Process:**
        1.  Takes the current `training_data` DataFrame (which includes initial positives and any "not\_[class]" examples from previous iterations).
        2.  Saves this `training_data` to a temporary `.pkl` file.
        3.  Constructs a command-line call to `train_representatives.py`, passing:
            *   Path to the temporary training data.
            *   Output directory for the current iteration.
            *   Number of epochs (from `PipelineConfig`).
            *   The margin to use for this specific iteration (primary or secondary, from `PipelineConfig`).
            *   `--no-auto-name` flag (assumed to tell the training script to use the provided output directory directly).
        4.  Executes this command as a subprocess.
        5.  Handles output (STDOUT, STDERR) and potential errors from the training script.
        6.  Returns the path to the `representatives.pkl` file generated by `train_representatives.py` for the current iteration.
    *   **Effect:** `train_representatives.py` learns representatives for all unique class labels present in its input data (e.g., "lamp", "mirror", "not\_lamp", "not\_mirror"). The "not\_[class]" examples influence the learned representatives for "lamp" and "mirror", ideally making them more robust by pushing them away from regions in the embedding space that correspond to previous false positives.

### 5. Prediction Generation: `iterative_pipeline/prediction_utils.py`

*   **`cosine_similarity_prediction()` function:**
    *   A basic utility that takes a single query embedding, a list of class representative embeddings, and their corresponding class names.
    *   Calculates the cosine similarity between the query embedding and each class representative.
    *   Returns the class name with the highest similarity and that similarity score.
*   **`generate_predictions()` function:**
    *   **Current Strategy:** Uses *only positive class representatives* for prediction.
    *   **Process:**
        1.  Loads the `representatives.pkl` file for the current iteration.
        2.  **Filters these representatives to keep only the "positive" classes** (e.g., "lamp", "mirror"), discarding any "not\_[class]" representatives. The "not\_" representatives are assumed to have already done their job during the training phase by refining the positive representatives.
        3.  For each embedding in the input `resnet_data` (full dataset for prediction):
            *   Calls `cosine_similarity_prediction` using only the filtered positive representatives.
        4.  Collects these predictions (`visual_predicted_object`, `visual_max_score`).
        5.  **Deduplicates predictions:** For each video and each (positive) predicted class, it keeps only the instance (frame) with the highest `visual_max_score`. This results in at most one prediction per class per video.
        6.  **Applies threshold:** Filters these deduplicated predictions based on the `current_iter_threshold` (primary or secondary).
        7.  Returns the final DataFrame of predictions for this iteration.

### 6. Evaluation & False Positive Extraction: `iterative_pipeline/evaluation_utils.py`

*   **`evaluate_predictions()` function:**
    *   Takes the thresholded, deduplicated predictions from `generate_predictions()` and the video-level `ground_truth` (from `trackingInfo`).
    *   For each video-class pair in the ground truth:
        *   Determines if the class was `actually present.
        *   Determines if the class was `predicted` (i.e., present in the input predictions for that video-class).
        *   Classifies the outcome as True Positive (TP), False Positive (FP), False Negative (FN), or True Negative (TN).
    *   Calculates and returns overall metrics (F1, Precision, Recall) and counts of TP, FP, FN, TN.
    *   Also returns a DataFrame (`evaluation_results`) detailing each video-class evaluation.
*   **`extract_false_positives()` function:**
    *   Takes the `evaluation_results` DataFrame and the `predictions_for_eval` DataFrame (the output of `generate_predictions` that was used for evaluation).
    *   Filters `evaluation_results` to find all FP cases.
    *   For each FP:
        *   Identifies the `video`, the `class` that was wrongly predicted as present, and the `frame` associated with that FP instance (from `evaluation_results`, which got it from `predictions_for_eval`).
        *   Looks up the original `finetuned_embedding` for this specific FP instance from the `predictions_for_eval` DataFrame.
        *   Creates a new training data entry: `{'class': 'not_[wrongly_predicted_class]', 'finetuned_embedding': [embedding_vector]}`.
    *   Returns a DataFrame (`fp_data`) of these new "not\_[class]" examples and a list of `new_exclusions` for tracking.

### 7. Single Iteration Management: `iterative_pipeline/iteration_manager.py`

*   **`run_single_iteration()` function:**
    *   Orchestrates the four main phases for a single iteration:
        1.  **Train:** Calls `train_reps_func` (which is `training_utils.train_contrastive_representatives`) with the current `training_data` and iteration-specific margin.
        2.  **Predict:** Calls `generate_preds_func` (which is `prediction_utils.generate_predictions`) using the newly trained representatives and the iteration-specific prediction threshold.
        3.  **Evaluate:** Calls `evaluate_preds_func` (which is `evaluation_utils.evaluate_predictions`).
        4.  **Extract FPs:** Calls `extract_fps_func` (which is `evaluation_utils.extract_false_positives`).
    *   Saves all artifacts for the iteration (representatives, evaluation CSVs, FP data, metrics JSON) into an iteration-specific subdirectory.
    *   **Updates Training Data:** Concatenates the `training_data` from the start of the iteration with the new `fp_data`. It performs deduplication to avoid adding identical "not\_[class]" examples (based on class label and embedding content using a tuple conversion for hashing).
    *   Returns the updated training data, metrics, and new exclusions for this iteration.

### 8. Overall Pipeline Orchestration: `iterative_pipeline/pipeline_manager.py`

*   **`run_iterative_pipeline()` function:**
    *   This is the main control loop for the entire iterative process.
    *   Initializes `current_training_data` with the `definitiveObjects`.
    *   Loops for the configured number of `iterations`:
        *   Determines the `current_iter_threshold_to_use` and `current_iter_margin_to_use` based on whether it's the first iteration (uses primary values from config) or a subsequent iteration (uses secondary values from config, if provided, otherwise defaults to primary).
        *   Calls `iteration_manager.run_single_iteration`, passing in all necessary data, the chosen parameters for the current iteration, and the actual functions for training, prediction, evaluation, and FP extraction (dependency injection).
        *   Updates `current_training_data` with the results from the iteration.
        *   Stores metrics and exclusion information.
        *   Checks for convergence: If the F1 score improvement over the previous iteration is below `config.convergence_threshold`, the loop breaks.
    *   After the loop, it saves a `pipeline_summary.json` with all configuration details and metrics from each iteration.
    *   Saves the `final_training_data.pkl`.

## IV. Data Flow Summary

1.  **Initial:** `definitiveObjects` (positive examples) -> Iteration 1 Training.
2.  **Iteration 1:**
    *   Train -> `reps_iter1.pkl`
    *   Predict (on `resnetPredictions`) -> `preds_iter1`
    *   Evaluate (`preds_iter1` vs `trackingInfo`) -> `eval_results_iter1` (get FPs)
    *   Extract FPs -> `fp_data_iter1` (e.g., "not\_lamp" examples)
3.  **Iteration 2:**
    *   New Training Data = `definitiveObjects` + `fp_data_iter1` (deduplicated)
    *   Train -> `reps_iter2.pkl` (representatives for "lamp", "mirror", "not\_lamp", etc. are learned; "lamp" and "mirror" reps are refined)
    *   Predict (on `resnetPredictions`, using *only refined positive reps* from `reps_iter2.pkl`) -> `preds_iter2`
    *   Evaluate -> `eval_results_iter2`
    *   Extract FPs -> `fp_data_iter2`
4.  ...and so on.

## V. Key Parameters and Their Roles

*   **`--threshold` / `--secondary-threshold`:** Control prediction confidence. The secondary threshold allows for a potentially different (e.g., lower) threshold after "not\_[class]" examples have been learned, which might help recover recall if the model becomes too conservative.
*   **`--margin` / `--secondary-margin`:** Control the separation enforced by the contrastive loss during training. The secondary margin allows tuning how aggressively the "not\_[class]" examples (and other negatives) push away the positive class representatives in later iterations.
*   **`--iterations`:** Maximum number of refinement cycles.
*   **`--convergence-threshold`:** Stops early if F1 score improvement becomes too small.

This refactored structure aims for better organization, making the complex iterative process easier to follow, debug, and extend.
