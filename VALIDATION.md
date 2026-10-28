# Archive validation

- Python syntax: 85 of 85 files passed.
- Credential-pattern scan: 0 flagged files (see local preparation report).
- Source files copied from existing projects; original folders were not modified.
- No training, LLM API calls or full research reproduction was performed.
- Private input datasets and generated model weights are excluded.
- The final EI AICc/Huber source and MSME three-way benchmark JSON were not located; see project READMEs.

Dependency lists added for chemical reactions and MSME are starting points inferred from imports, not verified lockfiles. Several original scripts require local path configuration and prior-stage outputs.

## Energy project addition

The energy notebook source (27 code cells, excluding its shell-install line) and preprocessing script passed syntax checks. Saved predictions were checked for finite numeric values; no credential patterns were found. Outputs were cleared from the published notebook; historical results are stored separately. No training was rerun. Known evaluation limitations are described in the project README.

## DSS project addition

Nine Python source files passed syntax checks. Six historical prediction tables were parsed and checked for finite numeric values; RMSE, MAE and R2 were recalculated from all saved predictions. No credential patterns were detected in the selected files. Full training and the live dashboard were not run. See the DSS README for scaling and evaluation-alignment limitations.
