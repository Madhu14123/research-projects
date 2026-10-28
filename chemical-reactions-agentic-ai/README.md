# Agentic AI for Chemical Reaction Refinement

Multi-agent reaction extraction/refinement with LangGraph, an MCP chemistry server, SMILES resolution and chemistry-aware evaluation.

## Main implementations

- `code/mas_reaction_pipeline_v3_publication_ready.py` with `code/chem_mcp_server_v3_publication_ready.py`: paired publication-ready source with saved ten-reaction outputs.
- `code/mas_reaction_pipeline_vf.py`: enhanced pipeline; its MCP server reference is **chem_mcp_server_v2.py**, which is included. The similarly named `chem_mcp_server_vf.py` is also retained as a historical variant.
- `code/mas_reaction_pipeline_v8.py` and `code/chem_mcp_server_v4.py`: later source variants; check the pipeline's server reference before choosing a variant.

## Setup

Install `requirements.txt` in a separate environment. Some optional chemistry features require additional external tools/services. The LLM setup tries OpenAI and then locally available Ollama models. Set `OPENAI_API_KEY` in your environment for OpenAI use; never put it in source code.

Before running, edit the bottom-of-file INPUT, OUTPUT_DIR and plots settings in the chosen pipeline. Point INPUT at `examples/first_10_jsons` and OUTPUT_DIR at a new local results directory. Run `python code/mas_reaction_pipeline_v3_publication_ready.py` from this project folder once paths and providers are configured.

## Saved evidence

`results/pipeline_output` and `results/pipeline_output_v3_publication_ready` contain existing metrics. Ten small reaction JSON examples are included instead of the original large reaction corpus. Saved metrics demonstrate historical runs, not a new evaluation or independent validation of chemical correctness. Some pipeline paths permit demo/fallback behaviour; inspect provider availability and output provenance when interpreting metrics.
