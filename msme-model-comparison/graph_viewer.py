from pathlib import Path
import json
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
GRAPH_JSON_CANDIDATES = [
    BASE_DIR / "outputs" / "msme_knowledge_graph_extensive.json",
    BASE_DIR / "outputs" / "msme_knowledge_graph.json",
]
DATASET_CSV = BASE_DIR / "MSME_new.csv"
SCHEMA_JSON_CANDIDATES = [
    BASE_DIR / "knowledge_graph_schema_extensive.json",
    BASE_DIR / "knowledge_graph_schema.json",
]

st.set_page_config(page_title="MSME Graph RAG Explorer", layout="wide")

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "show", "find", "list", "give",
    "give", "me", "to", "of", "in", "on", "into", "from", "who", "what", "where",
    "which", "how", "can", "you", "please", "need", "support", "related", "similar",
    "graph", "industry", "industries", "units", "entrepreneurs", "entrepreneur",
}


@st.cache_data(show_spinner=False)
def load_dataset():
    if DATASET_CSV.exists():
        return pd.read_csv(DATASET_CSV, low_memory=False)
    return None


@st.cache_data(show_spinner=False)
def load_schema():
    for path in SCHEMA_JSON_CANDIDATES:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return path, json.load(fh)
    return None, None


@st.cache_data(show_spinner=False)
def load_graph_data():
    for path in GRAPH_JSON_CANDIDATES:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return path, json.load(f)
    return None, None


def summarize_graph(graph):
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_type_counts = Counter(node.get("type") for node in nodes if node.get("type"))
    relation_counts = Counter(edge.get("relation") for edge in edges if edge.get("relation"))
    return nodes, edges, node_type_counts, relation_counts


def derive_question_taxonomy(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None:
        return {"dataset_columns": [], "question_types": []}

    columns = [col for col in df.columns.tolist() if col]
    question_types = []

    if "DISTRICT_NAME" in columns:
        question_types.append(
            {
                "area": "location",
                "question": "Which MSMEs are concentrated in a district or mandal?",
                "focus": "district and mandal segmentation",
            }
        )
    if "CATEGOERY_NAME" in columns:
        question_types.append(
            {
                "area": "sector",
                "question": "Which units belong to a given category or sector?",
                "focus": "category-based comparison",
            }
        )
    if "investment" in columns and "turnover" in columns:
        question_types.append(
            {
                "area": "finance",
                "question": "Which units show high investment but low turnover?",
                "focus": "financial performance and efficiency",
            }
        )
    if "employment" in columns:
        question_types.append(
            {
                "area": "workers",
                "question": "Which MSMEs are employment-heavy and relevant for worker support?",
                "focus": "job creation and workforce intensity",
            }
        )
    if "promoter_women" in columns or "no_of_females" in columns:
        question_types.append(
            {
                "area": "support",
                "question": "Which units are women-led or female-heavy and may need targeted support?",
                "focus": "women-led entrepreneurship and inclusion",
            }
        )
    if "PCBCATEGORY" in columns:
        question_types.append(
            {
                "area": "risk",
                "question": "Which units fall into higher compliance or environmental risk categories?",
                "focus": "risk screening and compliance",
            }
        )
    if "industrial_park" in columns:
        question_types.append(
            {
                "area": "supply_chain",
                "question": "Which firms are clustered inside industrial parks or estates?",
                "focus": "industrial clustering and linkage potential",
            }
        )
    if "ODOP" in columns:
        question_types.append(
            {
                "area": "policy",
                "question": "Which units are eligible for ODOP or sector-focused policy programs?",
                "focus": "program eligibility and intervention targeting",
            }
        )
    if "latitude" in columns and "longitude" in columns:
        question_types.append(
            {
                "area": "cluster",
                "question": "Which units form spatial clusters and nearby business ecosystems?",
                "focus": "geospatial clustering",
            }
        )

    question_types.append(
        {
            "area": "comparison",
            "question": "How do units compare across district, sector, risk, and employment dimensions?",
            "focus": "cross-cutting comparison",
        }
    )

    return {"dataset_columns": columns, "question_types": question_types}


def validate_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    if not isinstance(schema, dict):
        return {"is_valid": False, "issue_count": 1, "issues": ["Schema root is not a JSON object."]}

    required_top_level = ["graph_name", "node_types", "edge_types"]
    for key in required_top_level:
        if key not in schema:
            issues.append(f"Missing top-level key: {key}")

    node_types = schema.get("node_types", {})
    edge_types = schema.get("edge_types", {})
    if not isinstance(node_types, dict):
        issues.append("node_types must be an object")
    if not isinstance(edge_types, dict):
        issues.append("edge_types must be an object")

    if isinstance(node_types, dict):
        for node_name, node_meta in node_types.items():
            if not isinstance(node_meta, dict):
                issues.append(f"Node type {node_name} must be an object")
                continue
            if "properties" not in node_meta:
                issues.append(f"Node type {node_name} is missing properties")
            if "key_properties" not in node_meta:
                issues.append(f"Node type {node_name} is missing key_properties")

    if isinstance(edge_types, dict):
        for edge_name, edge_meta in edge_types.items():
            if not isinstance(edge_meta, dict):
                issues.append(f"Edge type {edge_name} must be an object")
                continue
            if edge_meta.get("from") not in node_types:
                issues.append(f"Edge type {edge_name} references unknown source node {edge_meta.get('from')}")
            if edge_meta.get("to") not in node_types:
                issues.append(f"Edge type {edge_name} references unknown target node {edge_meta.get('to')}")

    return {"is_valid": not issues, "issue_count": len(issues), "issues": issues}


def build_drilldown_subgraph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], root_node_id: str, depth: int = 2):
    if not root_node_id:
        return {"node_ids": [], "nodes": [], "edges": [], "depth": depth}

    adjacency = {}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source and target:
            adjacency.setdefault(source, []).append(target)
            adjacency.setdefault(target, []).append(source)

    visited = {root_node_id}
    queue = [(root_node_id, 0)]
    while queue:
        current_id, current_depth = queue.pop()
        if current_depth >= depth:
            continue
        for neighbor_id in adjacency.get(current_id, []):
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.append((neighbor_id, current_depth + 1))

    node_map = {node.get("id"): node for node in nodes if node.get("id")}
    selected_nodes = [node_map[node_id] for node_id in visited if node_id in node_map]
    selected_edges = [
        edge for edge in edges
        if edge.get("source") in visited and edge.get("target") in visited
    ]
    return {
        "node_ids": list(visited),
        "nodes": selected_nodes,
        "edges": selected_edges,
        "depth": depth,
    }


def build_type_summary_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
    node_map = {node.get("id"): node for node in nodes if node.get("id")}
    graph = nx.DiGraph()
    type_counts = Counter(node.get("type") for node in nodes if node.get("type"))
    for node_type, count in type_counts.items():
        graph.add_node(
            f"type:{node_type}",
            type="TypeSummary",
            label=f"{node_type} ({count})",
            properties={"count": count},
        )

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        source_type = node_map.get(source, {}).get("type")
        target_type = node_map.get(target, {}).get("type")
        if source_type and target_type:
            graph.add_edge(f"type:{source_type}", f"type:{target_type}", relation=edge.get("relation"))
    return graph


def export_artifacts():
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    dataset = load_dataset()
    taxonomy = derive_question_taxonomy(dataset)
    schema_path, schema = load_schema()
    validation = validate_schema(schema) if schema else {"is_valid": False, "issue_count": 1, "issues": ["Schema file was not found"]}

    with open(output_dir / "question_taxonomy.json", "w", encoding="utf-8") as fh:
        json.dump(taxonomy, fh, indent=2, ensure_ascii=False)
    with open(output_dir / "schema_validation_report.json", "w", encoding="utf-8") as fh:
        json.dump({"schema_file": str(schema_path.name) if schema_path else None, **validation}, fh, indent=2, ensure_ascii=False)

    return taxonomy, validation


def build_networkx(nodes, edges):
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(
            node["id"],
            type=node.get("type"),
            label=node.get("label"),
            properties=node.get("properties", {}),
        )
    for edge in edges:
        G.add_edge(edge.get("source"), edge.get("target"), relation=edge.get("relation"))
    return G


def get_node_color(node_type):
    palette = {
        "MSME": "#2563eb",
        "District": "#0f766e",
        "Mandal": "#7c3aed",
        "Category": "#ea580c",
        "RiskBucket": "#dc2626",
        "FinancialProfile": "#9333ea",
        "ScaleBand": "#16a34a",
        "GrowthSignal": "#ca8a04",
        "OperationalProfile": "#0891b2",
        "OwnershipPattern": "#be185d",
        "GeospatialCluster": "#475569",
    }
    return palette.get(node_type, "#9aa4b2")


def build_display_graph(nodes, edges, selected_type="All", selected_relation="All", max_nodes=220, max_edges=400):
    G = nx.DiGraph()
    visible_node_ids = {node["id"] for node in nodes if selected_type == "All" or node.get("type") == selected_type}

    for node in nodes:
        if node["id"] not in visible_node_ids:
            continue
        G.add_node(
            node["id"],
            type=node.get("type"),
            label=node.get("label"),
            properties=node.get("properties", {}),
        )

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in visible_node_ids or target not in visible_node_ids:
            continue
        if selected_relation != "All" and edge.get("relation") != selected_relation:
            continue
        G.add_edge(source, target, relation=edge.get("relation"))

    if G.number_of_nodes() > max_nodes:
        ranked_nodes = sorted(G.nodes(), key=lambda n: (G.degree(n), n), reverse=True)
        keep_nodes = set(ranked_nodes[:max_nodes])
        G = G.subgraph(keep_nodes).copy()

    if G.number_of_edges() > max_edges:
        ranked_edges = sorted(
            G.edges(data=True),
            key=lambda e: (G.degree(e[0]) + G.degree(e[1]), e[2].get("relation", "")),
            reverse=True,
        )
        keep_edges = {(e[0], e[1]) for e in ranked_edges[:max_edges]}
        G = G.edge_subgraph(keep_edges).copy()

    return G


def make_plotly_network(G, title, selected_nodes=None):
    if selected_nodes is None:
        selected_nodes = list(G.nodes())

    pos = nx.spring_layout(G, seed=42, k=0.7)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_hover = []
    for node in selected_nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_name = G.nodes[node].get("label") or node
        node_type = G.nodes[node].get("type", "")
        node_text.append(node_name)
        node_hover.append(f"{node_name}<br>{node_type}")
        node_color.append(get_node_color(node_type))
        node_size.append(min(24, 10 + 1.2 * G.degree(node)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.85, color="rgba(90,90,90,0.35)"),
            hoverinfo="skip",
            mode="lines",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(size=10, color="#111827"),
            marker=dict(size=node_size, color=node_color, line=dict(width=1.1, color="black")),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=node_hover,
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=700,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def parse_nl_query(query: str) -> Dict[str, Any]:
    text = (query or "").strip().lower()
    intent = "general"
    if any(token in text for token in ["women", "female", "woman", "women-led"]):
        intent = "women"
    elif any(token in text for token in ["cluster", "similar", "peer", "comparable", "group"]):
        intent = "cluster"
    elif any(token in text for token in ["supply", "chain", "link", "linkage", "connect", "connectivity"]):
        intent = "supply_chain"
    elif any(token in text for token in ["worker", "employment", "labor", "labour", "job", "jobs"]):
        intent = "workers"
    elif any(token in text for token in ["policy", "support", "recommend", "intervention", "program"]):
        intent = "policy"
    elif any(token in text for token in ["risk", "compliance", "environment", "pollution"]):
        intent = "risk"

    keywords = [
        token for token in re.findall(r"[a-zA-Z]+", text)
        if token.lower() not in STOPWORDS and len(token) > 2
    ]
    return {"intent": intent, "keywords": keywords, "query": query}


def _get_property(node: Dict[str, Any], prop: str, default=None):
    return node.get("properties", {}).get(prop, default)


def _build_graph_index(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
    node_map = {node["id"]: node for node in nodes}
    adjacency = {node_id: [] for node_id in node_map}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source and target:
            adjacency.setdefault(source, []).append((target, edge.get("relation")))
            adjacency.setdefault(target, []).append((source, edge.get("relation")))
    return node_map, adjacency


def _extract_anchor_ids(parsed: Dict[str, Any], node_map: Dict[str, Any], adjacency: Dict[str, List[Tuple[str, str]]]):
    keywords = parsed.get("keywords", [])
    anchor_ids = set()
    for node_id, node in node_map.items():
        node_type = node.get("type")
        if node_type == "MSME":
            continue
        label = (node.get("label") or "").lower()
        if any(keyword.lower() in label for keyword in keywords):
            anchor_ids.add(node_id)
    return anchor_ids


def _build_match_explanation(
    node: Dict[str, Any],
    props: Dict[str, Any],
    intent: str,
    adjacency: Dict[str, List[Tuple[str, str]]],
    node_map: Dict[str, Any],
    median_employment: float,
):
    context_nodes = []
    for neighbor_id, relation in adjacency.get(node["id"], []):
        neighbor = node_map.get(neighbor_id)
        if not neighbor or neighbor.get("type") == "MSME":
            continue
        context_nodes.append(f"{neighbor.get('type')}:{neighbor.get('label')}")

    evidence: List[str] = []
    reason_parts: List[str] = []

    employment = props.get("employment")
    risk_score = props.get("risk_score")
    female_percentage = props.get("female_percentage")
    is_women_led = bool(props.get("is_women_led", 0))
    is_odop = bool(props.get("is_odop_participant", 0))
    is_industrial_estate = bool(props.get("is_in_industrial_estate", 0))
    zero_employment_flag = bool(props.get("zero_employment_flag", 0))
    zero_turnover_flag = bool(props.get("zero_turnover_flag", 0))

    if intent == "women":
        if is_women_led:
            evidence.append("women-led flag")
            reason_parts.append("women-led ownership")
        if isinstance(female_percentage, (int, float)) and female_percentage >= 40:
            evidence.append(f"female share {female_percentage}%")
            reason_parts.append("strong female participation")
        if context_nodes:
            evidence.append("connected to support context nodes")
            reason_parts.append("graph context around support networks")
        if not reason_parts:
            reason_parts.append("profile similarity to the requested support pattern")
    elif intent == "cluster":
        if context_nodes:
            evidence.append("shared neighborhood with district/category/cluster entities")
            reason_parts.append("shared cluster context")
        if isinstance(employment, (int, float)) and employment >= median_employment:
            evidence.append("employment above median")
            reason_parts.append("meaningful scale")
        if not reason_parts:
            reason_parts.append("cluster-like graph connectivity")
    elif intent == "supply_chain":
        if is_industrial_estate:
            evidence.append("located in an industrial estate")
            reason_parts.append("industrial-estate connectivity")
        if isinstance(employment, (int, float)) and employment >= median_employment:
            evidence.append("employment above median")
            reason_parts.append("significant operating scale")
        if context_nodes:
            evidence.append("connected to linked industrial entities")
            reason_parts.append("supply-chain context")
        if not reason_parts:
            reason_parts.append("strong connectivity to industrial entities")
    elif intent == "workers":
        if isinstance(employment, (int, float)) and employment >= median_employment:
            evidence.append("employment above median")
            reason_parts.append("employment-intensive profile")
        if not zero_employment_flag and not zero_turnover_flag:
            evidence.append("active operating profile")
            reason_parts.append("active operations")
        if context_nodes:
            evidence.append("connected to local support context")
            reason_parts.append("local ecosystem relevance")
        if not reason_parts:
            reason_parts.append("workforce-relevant profile")
    elif intent == "policy":
        if isinstance(risk_score, (int, float)) and risk_score >= 0.5:
            evidence.append("risk score above policy threshold")
            reason_parts.append("higher policy relevance")
        if is_women_led:
            evidence.append("women-led flag")
            reason_parts.append("women-led support relevance")
        if is_odop:
            evidence.append("ODOP participation")
            reason_parts.append("ODOP-related relevance")
        if not reason_parts:
            reason_parts.append("policy-relevant graph context")
    else:
        if context_nodes:
            evidence.append("connected to nearby graph entities")
            reason_parts.append("relevant graph context")
        if isinstance(risk_score, (int, float)) and risk_score >= 0.5:
            evidence.append("risk score above threshold")
            reason_parts.append("risk-oriented relevance")
        if not reason_parts:
            reason_parts.append("general graph relevance")

    reason_sentence = "Matched because it shows " + ", ".join(reason_parts) + "."
    return reason_sentence, evidence, context_nodes


def _build_overall_explanation(parsed: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    intent = parsed.get("intent", "general")
    if intent == "women":
        base = "Why these results: the ranking prioritizes women-led or female-heavy MSMEs because the query asks for entrepreneurs who may need support."
    elif intent == "cluster":
        base = "Why these results: the ranking favors MSMEs that share stronger neighborhood and sectoral context because the question is about similar clusters and related industries."
    elif intent == "supply_chain":
        base = "Why these results: the ranking elevates firms with industrial-estate connectivity and nearby supply-chain context because the query asks for linkage opportunities."
    elif intent == "workers":
        base = "Why these results: the ranking prioritizes employment-intensive and operationally active units because the request focuses on worker-related opportunities."
    elif intent == "policy":
        base = "Why these results: the ranking highlights higher-risk or more policy-relevant enterprises because the question is about intervention design."
    else:
        base = "Why these results: the ranking uses graph evidence from the local neighborhood, risk signals, and support-related context to identify the best matches."

    evidence_summary = "Evidence used includes ownership signals, employment strength, risk score, and connected district/category/cluster nodes."
    if rows:
        top_reason = rows[0].get("why_it_matched", "")
        if top_reason:
            return f"{base} {evidence_summary} Top match rationale: {top_reason}"
    return f"{base} {evidence_summary}"


def execute_nl_query(query: str, graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    parsed = parse_nl_query(query)
    node_map, adjacency = _build_graph_index(nodes, edges)
    msme_nodes = [node for node in nodes if node.get("type") == "MSME"]

    if not msme_nodes:
        return {
            "intent": parsed["intent"],
            "summary": "No MSME nodes were found in the current graph.",
            "recommendations": [],
            "match_count": 0,
            "matches": [],
            "subgraph_nodes": [],
            "subgraph_edges": [],
        }

    anchor_ids = _extract_anchor_ids(parsed, node_map, adjacency)
    numeric_values = [
        _get_property(node, "employment") for node in msme_nodes if _get_property(node, "employment") is not None
    ]
    median_employment = float(pd.Series(numeric_values).median()) if numeric_values else 0.0

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for node in msme_nodes:
        props = node.get("properties", {})
        node_id = node["id"]
        score = 0.0
        label = node.get("label") or node_id
        text_label = label.lower()
        employment = props.get("employment")
        risk_score = props.get("risk_score")
        female_percentage = props.get("female_percentage")
        is_women_led = bool(props.get("is_women_led", 0))
        is_odop = bool(props.get("is_odop_participant", 0))
        is_industrial_estate = bool(props.get("is_in_industrial_estate", 0))
        zero_employment_flag = bool(props.get("zero_employment_flag", 0))
        zero_turnover_flag = bool(props.get("zero_turnover_flag", 0))

        if parsed["intent"] == "women":
            if is_women_led:
                score += 4.0
            if isinstance(female_percentage, (int, float)) and female_percentage >= 40:
                score += 2.0
            if any(keyword in text_label for keyword in ["women", "female", "sahaj", "self"]):
                score += 0.5
        elif parsed["intent"] == "cluster":
            shared_neighbors = 0
            for neighbor_id, _ in adjacency.get(node_id, []):
                if neighbor_id in anchor_ids:
                    shared_neighbors += 2
                elif node_map.get(neighbor_id, {}).get("type") in {"District", "Mandal", "Category", "IndustrialPark", "GeospatialCluster"}:
                    shared_neighbors += 1
            score += shared_neighbors
            if isinstance(employment, (int, float)) and employment >= median_employment:
                score += 0.5
        elif parsed["intent"] == "supply_chain":
            if is_industrial_estate:
                score += 2.0
            if isinstance(employment, (int, float)) and employment >= median_employment:
                score += 1.0
            if any(keyword in text_label for keyword in ["industry", "factory", "manufact", "textile", "food", "metal", "engineering"]):
                score += 0.5
        elif parsed["intent"] == "workers":
            if isinstance(employment, (int, float)) and employment >= median_employment:
                score += 3.0
            if not zero_employment_flag and not zero_turnover_flag:
                score += 1.0
        elif parsed["intent"] == "policy":
            if isinstance(risk_score, (int, float)) and risk_score >= 0.5:
                score += 3.0
            if is_women_led:
                score += 1.0
            if is_odop:
                score += 1.0
        elif parsed["intent"] == "risk":
            if isinstance(risk_score, (int, float)) and risk_score >= 0.5:
                score += 4.0

        if not score and parsed["intent"] == "general":
            score = len(adjacency.get(node_id, [])) / 10.0

        scored.append((score, node))

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    selected_nodes = [node for _, node in ranked[:18]]

    selected_ids = {node["id"] for node in selected_nodes}
    for node in selected_nodes:
        for neighbor_id, _ in adjacency.get(node["id"], []):
            selected_ids.add(neighbor_id)

    subgraph = nx.DiGraph()
    for node_id in selected_ids:
        node = node_map.get(node_id)
        if node is None:
            continue
        subgraph.add_node(
            node_id,
            type=node.get("type"),
            label=node.get("label"),
            properties=node.get("properties", {}),
        )

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in selected_ids and target in selected_ids:
            subgraph.add_edge(source, target, relation=edge.get("relation"))

    rows = []
    for node in selected_nodes:
        props = node.get("properties", {})
        why_it_matched, evidence, context_nodes = _build_match_explanation(
            node,
            props,
            parsed["intent"],
            adjacency,
            node_map,
            median_employment,
        )
        rows.append(
            {
                "id": node["id"],
                "name": props.get("UNIT_NAME") or node.get("label"),
                "type": node.get("type"),
                "employment": props.get("employment"),
                "risk_score": props.get("risk_score"),
                "female_percentage": props.get("female_percentage"),
                "women_led": bool(props.get("is_women_led", 0)),
                "context": "; ".join(context_nodes[:4]),
                "why_it_matched": why_it_matched,
                "evidence": "; ".join(evidence[:4]),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        top_context = df["context"].fillna("").astype(str)
        top_context = top_context[top_context != ""]
    else:
        top_context = pd.Series(dtype="object")

    summary_lines = []
    if parsed["intent"] == "women":
        summary_lines.append(
            f"Found {len(rows)} women-focused MSME candidates for targeted support."
        )
        summary_lines.append(
            "These units are strong candidates for women-led credit, mentoring, procurement access, and market linkage programs."
        )
    elif parsed["intent"] == "cluster":
        summary_lines.append(
            f"Found {len(rows)} MSMEs that form a strong local or sectoral cluster pattern."
        )
        summary_lines.append(
            "These units are suitable for cluster-based infrastructure, shared services, and supplier-network development."
        )
    elif parsed["intent"] == "supply_chain":
        summary_lines.append(
            f"Found {len(rows)} MSMEs with strong supply-chain linkage potential."
        )
        summary_lines.append(
            "These units can be connected through shared industrial parks, categories, and buyer-supplier ecosystems."
        )
    elif parsed["intent"] == "workers":
        summary_lines.append(
            f"Found {len(rows)} labor-intensive or employment-significant MSMEs."
        )
        summary_lines.append(
            "These units are relevant for worker welfare, skilling, job quality, and employment-focused policy support."
        )
    elif parsed["intent"] == "policy":
        summary_lines.append(
            f"Found {len(rows)} MSMEs that warrant targeted policy attention."
        )
        summary_lines.append(
            "These units are useful for designing targeted finance, compliance, and capability-building interventions."
        )
    else:
        summary_lines.append(f"Found {len(rows)} MSMEs that match the query context.")
        summary_lines.append("The retrieved subgraph highlights the closest connected entities and policy-relevant relationships.")

    if not top_context.empty:
        summary_lines.append(f"Common context anchors: {', '.join(top_context.dropna().astype(str).head(3))}")

    recommendations = []
    if parsed["intent"] == "women":
        recommendations.extend([
            "Create women-led credit and guarantee schemes for the matched units.",
            "Offer procurement and market-linkage support through public and private buyer networks.",
            "Provide mentoring and digital-skills support for women founders and operators.",
        ])
    elif parsed["intent"] == "cluster":
        recommendations.extend([
            "Support cluster-based shared facilities, testing labs, and logistics services.",
            "Encourage supplier aggregation and collaborative purchasing to strengthen local value chains.",
            "Use district-level industrial planning to connect similar MSMEs into compact ecosystems.",
        ])
    elif parsed["intent"] == "supply_chain":
        recommendations.extend([
            "Create buyer-supplier matchmaking programs between complementary industries.",
            "Identify shared infrastructure and transport bottlenecks for clustered units.",
            "Encourage shared services such as warehousing, packaging, and certification.",
        ])
    elif parsed["intent"] == "workers":
        recommendations.extend([
            "Prioritize worker training, safety, and welfare support for the matched units.",
            "Design labor-intensive credit and production support programs for these firms.",
            "Link these units with skill-development and apprenticeship schemes.",
        ])
    elif parsed["intent"] == "policy":
        recommendations.extend([
            "Use the retrieved firms as pilot cases for compliance and capability programs.",
            "Pair finance support with risk-reduction services such as audits and advisory support.",
            "Target district-level policy packages around the strongest recurring themes in the result set.",
        ])
    else:
        recommendations.extend([
            "Use the retrieved cluster as a policy prototype for district-level industrial support.",
            "Compare these firms on risk, employment, and women-led characteristics to design better interventions.",
        ])

    return {
        "intent": parsed["intent"],
        "summary": "\n".join(summary_lines),
        "explanation": _build_overall_explanation(parsed, rows),
        "reasoning_steps": [
            "The query intent was classified from the user question.",
            "The graph was traversed to gather nearby entities and structural context.",
            "Each match was ranked using ownership, employment, risk, and relationship evidence.",
        ],
        "recommendations": recommendations,
        "match_count": len(rows),
        "matches": rows,
        "subgraph_nodes": list(subgraph.nodes()),
        "subgraph_edges": [
            {"source": source, "target": target, "relation": data.get("relation")}
            for source, target, data in subgraph.edges(data=True)
        ],
    }


def main():
    st.title("MSME Graph RAG Explorer")
    st.caption("Start with a high-level view, then drill down into a node to see its local relationships and relevant context.")

    taxonomy, schema_validation = export_artifacts()
    path, graph = load_graph_data()
    if path is None or graph is None:
        st.error(
            "No graph JSON found. Run the graph builder first to create one of:\n"
            + "\n".join(str(p) for p in GRAPH_JSON_CANDIDATES)
        )
        return

    nodes, edges, node_type_counts, relation_counts = summarize_graph(graph)

    st.subheader("Question types the dataset can support")
    st.caption("These are the question families that the current dataset can answer before we move to a full knowledge graph implementation.")
    for item in taxonomy.get("question_types", []):
        st.write(f"• {item['question']} ({item['focus']})")

    st.subheader("Schema validation")
    if schema_validation.get("is_valid"):
        st.success("The current schema is structurally valid.")
    else:
        st.warning("The current schema still needs attention before the graph is expanded.")
    if schema_validation.get("issues"):
        for issue in schema_validation["issues"]:
            st.write(f"• {issue}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", len(nodes))
    col2.metric("Edges", len(edges))
    col3.metric("Graph file", path.name)

    st.subheader("High-level to low-level explorer")
    st.caption("The first layer is a coarse view of entity types; the second layer expands around a selected node to reveal its nearby relationships.")

    overview_graph = build_type_summary_graph(nodes, edges)
    overview_fig = make_plotly_network(overview_graph, "High-level view by node type")
    st.plotly_chart(overview_fig, use_container_width=True)

    candidate_nodes = [node for node in nodes if node.get("id")]
    candidate_options = [
        f"{node.get('label') or node.get('id')} ({node.get('type') or 'unknown'})"
        for node in candidate_nodes[:120]
    ]
    if candidate_options:
        selected_option = st.selectbox("Select a node to drill down", candidate_options, index=0)
        selected_node_id = candidate_nodes[candidate_options.index(selected_option)].get("id")
        drilldown = build_drilldown_subgraph(nodes, edges, selected_node_id, depth=2)
        drilldown_graph = build_networkx(drilldown["nodes"], drilldown["edges"])
        drilldown_fig = make_plotly_network(drilldown_graph, f"Drill-down around {selected_node_id}")
        st.plotly_chart(drilldown_fig, use_container_width=True)

    with st.sidebar:
        st.subheader("Graph RAG Query")
        example_queries = [
            "find women entrepreneurs who need support",
            "find similar clusters and related industries",
            "show supply chain linkages between industries",
            "find workers and employment-heavy MSMEs",
            "design policy support for high-risk firms",
        ]
        st.caption("Try one of these examples or enter your own question.")
        for example in example_queries:
            st.code(example, language="text")
        query_input = st.text_area(
            "Ask in plain English",
            value="find women entrepreneurs who need support",
            height=90,
        )
        run_query = st.button("Run Graph RAG Query")

    if run_query:
        with st.spinner("Retrieving context from the graph..."):
            result = execute_nl_query(query_input, graph)
        st.subheader("Query result")
        st.info(result["summary"])

        st.subheader("Why these results")
        st.write(result["explanation"])
        with st.expander("Reasoning trace"):
            for step in result["reasoning_steps"]:
                st.write(f"• {step}")

        if result["recommendations"]:
            st.subheader("Recommended policy and program actions")
            for rec in result["recommendations"]:
                st.write(f"• {rec}")

        if result["matches"]:
            st.subheader("Matched MSMEs")
            match_df = pd.DataFrame(result["matches"])
            display_cols = [
                "name",
                "employment",
                "risk_score",
                "female_percentage",
                "women_led",
                "context",
                "why_it_matched",
                "evidence",
            ]
            st.dataframe(match_df[[col for col in display_cols if col in match_df.columns]], use_container_width=True)

            if result["subgraph_nodes"]:
                subgraph = nx.DiGraph()
                node_map = {node["id"]: node for node in nodes}
                for node_id in result["subgraph_nodes"]:
                    node = node_map.get(node_id)
                    if node is None:
                        continue
                    subgraph.add_node(
                        node_id,
                        type=node.get("type"),
                        label=node.get("label"),
                        properties=node.get("properties", {}),
                    )
                for edge in result["subgraph_edges"]:
                    subgraph.add_edge(edge.get("source"), edge.get("target"), relation=edge.get("relation"))
                fig = make_plotly_network(subgraph, f"Graph RAG context for: {query_input}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No matching MSMEs were found for that query.")

    st.subheader("Readable graph view")
    view_mode = st.radio(
        "Choose graph view",
        ["Focused overview", "Neighborhood explorer"],
        horizontal=True,
    )

    if view_mode == "Focused overview":
        node_types = ["All"] + sorted(node_type_counts.keys())
        relation_types = ["All"] + sorted(relation_counts.keys())
        selected_type = st.selectbox("Filter by node type", node_types, index=0)
        selected_relation = st.selectbox("Filter by relation", relation_types, index=0)
        max_nodes = st.slider("Max nodes to display", min_value=60, max_value=300, step=20, value=180)
        max_edges = st.slider("Max edges to display", min_value=80, max_value=800, step=20, value=260)

        display_graph = build_display_graph(
            nodes,
            edges,
            selected_type=selected_type,
            selected_relation=selected_relation,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

        st.caption(
            f"Showing {display_graph.number_of_nodes()} nodes and {display_graph.number_of_edges()} edges for a clearer view."
        )
        fig = make_plotly_network(display_graph, "Focused overview of the knowledge graph")
        st.plotly_chart(fig, use_container_width=True)
    else:
        base_graph = build_networkx(nodes, edges)
        degree_sorted_nodes = [node_id for node_id, _ in sorted(base_graph.degree(), key=lambda item: (item[1], item[0]), reverse=True)[:80]]
        root_node = st.selectbox("Select a root node", degree_sorted_nodes, index=0)
        depth = st.slider("Neighborhood depth", min_value=1, max_value=3, value=2)
        try:
            subgraph = nx.ego_graph(base_graph, root_node, radius=depth, undirected=False)
            fig = make_plotly_network(subgraph, f"Neighborhood around {root_node}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Neighborhood view could not be generated: {exc}")

    st.subheader("Graph composition")
    type_chart = px.bar(
        x=list(node_type_counts.keys()),
        y=list(node_type_counts.values()),
        labels={"x": "Node type", "y": "Count"},
        title="Node type distribution",
    )
    st.plotly_chart(type_chart, use_container_width=True)

    rel_chart = px.bar(
        x=list(relation_counts.keys()),
        y=list(relation_counts.values()),
        labels={"x": "Relation", "y": "Count"},
        title="Relation distribution",
    )
    st.plotly_chart(rel_chart, use_container_width=True)

    st.subheader("Node details")
    node_df = pd.DataFrame(
        [
            {
                "id": node.get("id"),
                "type": node.get("type"),
                "label": node.get("label"),
                "properties": json.dumps(node.get("properties", {}), ensure_ascii=False)[:300],
            }
            for node in nodes
        ]
    )
    st.dataframe(node_df, use_container_width=True)


if __name__ == "__main__":
    main()
