# Agentic RAG: Research-Focused AI Assistant

An experimental assistant for exploring AI research articles. It retrieves relevant passages, checks their relevance to the question, and uses retrieved text to generate an answer.

## Source

Copied from [Madhu14123/Agentic_RAG](https://github.com/Madhu14123/Agentic_RAG), notebook blob `d452cc52f0a7edeb5e40ef946f8640d50b03620a`. The original repository remains unchanged.

## Workflow

- Load two public Lilian Weng articles about LLM agents and prompt engineering.
- Split article text into overlapping chunks.
- Embed chunks using Hugging Face `all-MiniLM-L6-v2` and index them in Chroma.
- Use a LangGraph agent to decide whether to retrieve text or answer directly.
- Grade retrieved text for relevance, generate an answer when relevant, or rewrite the question and return to the agent.

The notebook includes Gemini experiments and a later Groq Gemma2 configuration. It uses LangChain, LangGraph, Pydantic, Chroma and Hugging Face embeddings.

## Running

Open `agentic_rag.ipynb` in Jupyter. Dependencies inferred from imports include langchain, langchain-core, langchain-community, langchain-text-splitters, langgraph, langchain-google-genai, langchain-huggingface, langchain-groq, chromadb, sentence-transformers, python-dotenv, beautifulsoup4, tiktoken and ipython. These are not a tested lockfile.

Configure the environment variables used in the first cells and review model availability before running. Two later setup lines currently assign empty Groq and Hugging Face credentials; replace those assignments with your local environment configuration. Do not commit credentials.

## Scope and verification

This is a notebook prototype, not a validated research search product. It was demonstrated on two web articles, rather than a broad paper collection. Its rewrite path returns to an assistant branch that may answer directly instead of retrieving again, so repeated retrieval is not guaranteed. No accuracy benchmark or fresh live model run is claimed.

For this portfolio copy, saved outputs and execution counts were cleared, cell metadata was removed, and one prose-only code cell was changed to Markdown. The 57 remaining code cells passed Python syntax checks with notebook install commands excluded. No credential patterns were detected in the cleaned source. The notebook contains explanatory notes and exploratory cells; run it with appropriate current dependencies and API access.
