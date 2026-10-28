# MSME Knowledge Graph Project Analysis

## What has been done so far

### 1. Data preparation
- The project ingests the MSME dataset from outputs/MSME_preprocessed_full.csv.
- The preprocessing pipeline standardizes fields and creates a richer feature set for downstream analysis.

### 2. Knowledge graph construction
- Two graph builders are available:
  - build_knowledge_graph.py for a baseline graph.
  - build_knowledge_graph_extensive.py for a richer graph with derived risk, financial, operational, ownership, and geospatial nodes.
- The resulting graph is exported as JSON and GraphML for downstream visualization and querying.

### 3. Exploratory analysis and modeling
- eda_patterns.py produces summary statistics and visual patterns.
- build_models_pipeline.py trains baseline models for clustering, environmental risk, women-led MSME performance, and credit readiness.
- next_phase_models.py and phase2_next_steps.py extend the analysis toward clustering and policy-oriented next steps.

### 4. Visualization and interaction
- graph_viewer.py now provides a readable knowledge graph explorer.
- The viewer includes:
  - a focused overview view,
  - a neighborhood explorer,
  - a graph-RAG style English query interface.

### 5. Outputs already generated
- The outputs folder contains preprocessed data, graph JSON/GraphML exports, model metrics, and reports.
- This provides a base for policy support, cluster analysis, worker-focused planning, and supply-chain exploration.

## Where the new Graph RAG capability helps

The new English query layer supports questions such as:
- women entrepreneurs and women-led support needs,
- similar clusters and related industries,
- supply-chain linkages and industrial connectivity,
- workers and employment-focused opportunities,
- policy interventions for high-risk or underserved MSMEs.

This turns the graph from a static topology into a policy and decision-support tool.
