# Emotional Intelligence Research

Computational modelling of emotional intelligence during entrance-examination problem solving. The source study uses facial action units, manually coded stages, SAM self-reports and exam outcomes for 42 participants.

## Thesis

[Read the master's thesis (PDF)](thesis/masters-thesis.pdf) — *EI in the Age of AI: A measurement framework for emotional intelligence in Indian competitive exams*, Uppara Madhurya Sri, Blekinge Institute of Technology.

The selected 114-page local version has a May 2026 cover date and a PDF creation date of 13 October 2026. It is the latest complete matching thesis found locally; its identity as the exact submitted copy is not independently confirmed. See [version details](thesis/README.md). The PDF is unchanged.

## Main source groups

- `src/run_pipeline.py`: modular five-stage loader, feature construction, composite, validation and confirmatory analysis. Additional stage modules provide regression, temporal and multitask experiments; the runner itself runs stages 1-5.
- `code/ei_candidate_features.py`, `ei_features_from_pdf.py` and `ei_feature_pipeline_v5.py`: feature-engineering implementations.
- `code1/feature_selection_and_model_comparison.py`: seven-method voting and comparison of 14 regression models across feature counts.
- `code2/consensus_feature_pool.py`: consensus-pool selection; related interaction, ensemble and diagnostic scripts are retained beside it.
- `temporary1/interaction_only_model.py`, `interaction_only_expanded.py`, `interaction_only_max.py`: interaction-based forward-selection experiments.
- `temporary1/ei_final_scoring_v2.py`: corrected 15-feature Ridge scoring model with a saved report in `results/results4/ei_final_report_v2.txt`.
- `code/`: selected feature, temporal, Markov, statistical and deep-learning comparisons supporting the research.

## Setup and inputs

Use a separate Python environment. Start with `code/requirements.txt`; `code/requirements_dl.txt` covers deep-learning experiments. Some scripts additionally require packages such as lightgbm, kneed or factor-analyzer; consult their imports.

Original scripts refer to `D:/A_emotions in exam` and related local locations. Configure the paths in the selected script and in `src/config.py` before running. The original raw input folder is currently absent from the located project. Required inputs include facial AU CSVs, stage annotations, `Exam Scores.xlsx` and SAM responses. Later scripts consume `features_D1.csv` through `features_D5.csv` and other earlier-stage outputs. These participant-level inputs are deliberately excluded.

From this folder, run the modular workflow with `python src/run_pipeline.py` after configuring the paths and supplying authorised inputs. Run other scripts only after checking their input/output path settings.

## Saved evidence and thesis coverage

Selected aggregate model tables, feature-selection histories and summaries are retained under `results/`. The corrected Ridge report records LOOCV r=0.8949 with 15 features. These are historical saved results.

The located `D:/A_TEMPORARY/thesis_final.tex` describes a later 23-configuration AICc comparison with a 12-feature HuberRegressor (AICc 139.44, r=0.9755). The exact source implementation and matching result tables for that final comparison were **not found** in the searched D: drive research source files. This archive includes related feature-voting and interaction implementations, but does not claim to reproduce that final thesis result. The thesis itself describes the reported correlation as exploratory because interaction screening preceded LOOCV.

Participant scores, facial measurements, per-participant feature matrices and figures containing participant-level results are excluded.
