# Archive validation

- Initial archive Python syntax check: 85 of 85 files passed.
- Credential-pattern scan: 0 flagged files (see local preparation report).
- Source files copied from existing projects; original folders were not modified.
- No training, LLM API calls or full research reproduction was performed.
- Private input datasets and generated model weights are excluded.
- The final EI AICc/Huber source and MSME three-way benchmark JSON were not located; see project READMEs.

Dependency lists added for chemical reactions and MSME are starting points inferred from imports, not verified lockfiles. Several original scripts require local path configuration and prior-stage outputs.

## majorpro4 revision

The revised CSV-to-SQL app passed 16 unit tests and a Streamlit AppTest smoke check covering startup, a demo aggregate, rejected mutation and intact data. Live model inference and a fresh dependency installation were not performed. Original credential-bearing variants, databases and survey records were excluded; synthetic demo data is included.

## Agentic RAG addition

Copied from the separate Agentic_RAG repository. Cleared saved outputs and execution counts, removed cell metadata, and changed one prose-only code cell to Markdown. The remaining 57 code cells passed syntax checks with notebook install commands excluded. Credential-pattern scan found no matches. No live API calls, fresh dependency installation or retrieval-quality benchmark was performed. See the project guide for prototype limitations.
