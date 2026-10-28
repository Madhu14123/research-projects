# Natural Language to SQL: CSV Explorer

A revised version of the local `majorpro4` project for asking questions about CSV data with local Hugging Face models, reviewing generated SQL and executing read-only SQLite queries in a Streamlit interface.

## Features

- Upload a CSV or use the included synthetic food-preference survey demo.
- Inspect a schema derived from the actual uploaded columns, types and example values.
- Choose the original project's T5 WikiSQL approach or a Llama 2 Chat model.
- Load/cache a model only when Generate SQL is selected.
- Review/edit the generated SQL before running it. The SQL editor also works without downloading a model.
- Execute one read-only query against an isolated in-memory table with a row cap, function restrictions and a query time budget.
- Inspect results in a table, including valid queries that return no rows.

## Setup and run

Use Python 3.10 or 3.11 with a separate virtual environment and sufficient disk/RAM for the selected model.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run from this project folder. The demo query works immediately using the synthetic CSV. No API key or model download is needed to edit and run SQL manually.

For natural-language generation, enter a question and choose Generate SQL. The first call downloads model files from Hugging Face; this can take time and substantial disk/RAM. T5 is the smaller option. Llama 2 requires acceptance of the model's access terms, substantial memory, and an authorised `HF_TOKEN` environment variable. `.env.example` documents its name; the app reads the process environment and does not automatically load `.env` files. Never put credentials in source code.

The requirements are compatible version ranges, not a frozen environment. Full installation on a new machine and real model inference have not been tested during this revision.

## Example questions

- How many respondents find the food tasty?
- What is the average age by gender?
- What is the most common visit frequency?

The table name is always `mcdonalds_data`, including for an uploaded CSV. Use the actual uploaded column names. Quote identifiers such as `"Like"` with double quotes. Generated SQL is probabilistic: the pretrained WikiSQL model is not fine-tuned for this dataset, and queries may need correction. This project does not claim a measured text-to-SQL accuracy or fine-tuning result.

## Changes from the original project

The local project had several T5, Llama and Gemini variants. This revision consolidates the T5/Llama workflow and fixes database lifecycle issues, schema mismatches, generation-length handling and prompt-inclusive Llama decoding. It removes ad hoc string substitutions for SQL column/value guessing and replaces unrestricted execution with read-only SQLite authorization and bounded results. Generation is deterministic and separate from query execution.

Some earlier Gemini variants contained hard-coded credentials. They are excluded, along with the original databases, survey records and model weights. The included demo rows are synthetic. Original files under `D:/majorpro4` were not modified. `PROVENANCE.json` records which original files informed the revision. This is a documented revision, not a claim that the original project contained these fixes.

## Verification

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

Sixteen tests passed, covering aggregates, quoted identifiers, CTEs, row limits, empty results, fenced SQL, write/external-access rejection, timeout handling, schema prompts and causal-versus-sequence-to-sequence decoding using model doubles. Streamlit AppTest also verified startup, the demo result, rejected mutation and unchanged data afterwards. The model-double tests do not verify actual model output quality. No real model was downloaded and no live inference benchmark was performed.

The query executor is a limited local CSV explorer, not a general-purpose database-access service. The timeout applies to SQLite execution, not model generation or CSV ingestion. A 10 MB CSV upload limit is applied in the interface.

## Implementation references

- [Hugging Face generation API](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Python SQLite callbacks](https://docs.python.org/3.13/library/sqlite3.html)
- [SQLite authorization callbacks](https://www.sqlite.org/c3ref/set_authorizer.html)
