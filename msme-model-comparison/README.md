# MSME Model Comparison and Knowledge Graphs

Research code for MSME preprocessing, exploratory analysis, clustering, risk/readiness models, knowledge graphs, dashboards and natural-language query modelling.

## Main workflows

1. `pipeline.py` and `convert_data.py`: data preprocessing; inspect expected input names before running. The original input is `MSME_new.csv`. `data/EXPECTED_COLUMNS.json` preserves its column names without raw records.
2. `build_models_pipeline.py`: KMeans clustering and baseline classification with saved metrics under `outputs/build_models/`.
3. `next_phase_models.py`, `phase2_next_steps.py` and `train_*_model.py`: later clustering, environmental-risk, credit and women-led business analyses. Saved model metrics and reports are included under `outputs/phase2/` and `outputs/phase2_next_steps/`.
4. `build_knowledge_graph.py`, `build_knowledge_graph_extensive.py`, `graph_viewer.py` and `msme_dashboard.py`: graph construction, querying and visualisation, with graph schemas and tests.
5. `ml_query/train_all_models.py`: shared-test-set comparison of TF-IDF + Logistic Regression, sentence-transformer features and T5-small. `train_t5_only.py` is the separate T5 training entry point. Synthetic query generation and serving code are included.

## Setup

Install root `requirements.txt` in a separate environment. Query-model experiments have additional dependencies in `ml_query/requirements.txt`. These lists are derived from source imports and are not an environment lockfile.

Scripts often use the current working directory. Run from this project folder and inspect each script's input/output settings. Supply authorised `MSME_new.csv` / `outputs/MSME_preprocessed_full.csv` locally before running `python build_models_pipeline.py` or the related scripts. Input databases and generated graph files must be created before using the viewers.

For query-model comparison, inspect `ml_query/generate_training_data.py`, generate `train.json` and `test.json` as expected by the trainer, then run `python ml_query/train_all_models.py`. Two locally saved classifier files and a T5 model directory were found; they are excluded as generated weights. The trainer's expected `benchmark_results.json` was not present, so no completed three-way benchmark score is claimed.

## Dashboard source

`msme-dashboard/` preserves the Vite/React application source and dependency lockfile. Dataset-bearing generated JavaScript/JSON files were excluded. Regenerate its data and geographic assets using the supplied generation code and authorised local inputs before building. It is not a standalone ready-to-run dashboard without those assets.

## Interpretation

`outputs/` contains historical metrics/reports supporting completed modelling runs. These are not newly reproduced scores. Several tasks use derived or rule-based labels; inspect label construction and feature overlap before interpreting high accuracy as predictive validity. Raw company records, personal information, databases, generated record-level assignments and trained model weights are excluded.
