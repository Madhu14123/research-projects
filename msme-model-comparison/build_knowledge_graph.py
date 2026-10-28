from pathlib import Path
import json
import sqlite3
import warnings
from typing import Dict, Any

import networkx as nx
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path.cwd()
CSV_PATH = BASE_DIR / "outputs" / "MSME_preprocessed_full.csv"
SCHEMA_PATH = BASE_DIR / "knowledge_graph_schema.json"
GRAPH_DB = BASE_DIR / "outputs" / "msme_knowledge_graph.db"
GRAPH_JSON = BASE_DIR / "outputs" / "msme_knowledge_graph.json"
GRAPH_GRAPHML = BASE_DIR / "outputs" / "msme_knowledge_graph.graphml"


def load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(val):
    if pd.isna(val):
        return None
    return str(val).strip()


def dedupe_id(prefix: str, value: Any):
    return f"{prefix}:{normalize(value)}"


def create_graph_db(df: pd.DataFrame, schema: Dict[str, Any]):
    conn = sqlite3.connect(GRAPH_DB)
    cursor = conn.cursor()

    cursor.executescript(
        """
        DROP TABLE IF EXISTS kg_nodes;
        DROP TABLE IF EXISTS kg_edges;
        CREATE TABLE kg_nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            label TEXT,
            properties_json TEXT
        );
        CREATE TABLE kg_edges (
            edge_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            relation TEXT NOT NULL,
            target TEXT NOT NULL,
            properties_json TEXT
        );
        """
    )

    # Create graph object
    G = nx.DiGraph()

    # Add nodes and edges
    for _, row in df.iterrows():
        m_id = dedupe_id("MSME", row.get("SL", row.get("UNIT_NAME")))
        G.add_node(
            m_id,
            node_type="MSME",
            label=normalize(row.get("UNIT_NAME")),
            unit_name=normalize(row.get("UNIT_NAME")),
            sl=normalize(row.get("SL")),
            investment=row.get("investment"),
            employment=row.get("employment"),
            turnover=row.get("turnover"),
            risk_score=row.get("risk_score"),
            female_percentage=row.get("female_percentage"),
            roi_ratio=row.get("ROI_ratio"),
            revenue_per_employee=row.get("revenue_per_employee")
        )

        # Entity nodes and edges
        for entity_field, entity_type in [
            ("DISTRICT_NAME", "District"),
            ("mandal_name", "Mandal"),
            ("CATEGOERY_NAME", "Category"),
            ("industrial_park", "IndustrialPark"),
            ("PCBCATEGORY", "PCBCategory"),
            ("social_status", "SocialStatus"),
            ("promoter_women", "PromoterType"),
            ("ODOP", "ODOPFlag"),
        ]:
            value = normalize(row.get(entity_field))
            if value is None:
                continue

            entity_id = dedupe_id(entity_type, value)
            G.add_node(entity_id, node_type=entity_type, label=value)
            relation_map = {
                "DISTRICT_NAME": "LOCATED_IN_DISTRICT",
                "mandal_name": "LOCATED_IN_MANDAL",
                "CATEGOERY_NAME": "BELONGS_TO_CATEGORY",
                "industrial_park": "HAS_INDUSTRIAL_PARK",
                "PCBCATEGORY": "HAS_PCB_CATEGORY",
                "social_status": "HAS_SOCIAL_STATUS",
                "promoter_women": "HAS_PROMOTER_TYPE",
                "ODOP": "PARTICIPATES_IN_ODOP"
            }
            G.add_edge(m_id, entity_id, relation=relation_map.get(entity_field, "RELATED_TO"))

        # Risk bucket
        risk_bucket = None
        if row.get("risk_score") is not None:
            if row.get("risk_score") >= 0.7:
                risk_bucket = "HIGH_RISK"
            elif row.get("risk_score") >= 0.4:
                risk_bucket = "MEDIUM_RISK"
            else:
                risk_bucket = "LOW_RISK"
        if risk_bucket:
            risk_id = dedupe_id("RiskBucket", risk_bucket)
            G.add_node(risk_id, node_type="RiskBucket", label=risk_bucket)
            G.add_edge(m_id, risk_id, relation="HAS_RISK_BUCKET")

    # Save nodes and edges to SQLite
    for node_id, attrs in G.nodes(data=True):
        cursor.execute(
            "INSERT OR REPLACE INTO kg_nodes(node_id, node_type, label, properties_json) VALUES (?, ?, ?, ?)",
            (
                node_id,
                attrs.get("node_type"),
                attrs.get("label"),
                json.dumps({k: v for k, v in attrs.items() if k not in ("node_type", "label")}, default=str)
            )
        )

    for source, target, attrs in G.edges(data=True):
        edge_id = f"{source}::{attrs.get('relation')}::{target}"
        cursor.execute(
            "INSERT OR REPLACE INTO kg_edges(edge_id, source, relation, target, properties_json) VALUES (?, ?, ?, ?, ?)",
            (edge_id, source, attrs.get("relation"), target, json.dumps(attrs, default=str))
        )

    conn.commit()
    conn.close()

    # Export JSON and GraphML for downstream tools
    nx.write_graphml(G, GRAPH_GRAPHML)

    graph_json = {
        "nodes": [
            {"id": node_id, "type": data.get("node_type"), "label": data.get("label"), "properties": data.get("properties", {})}
            for node_id, data in G.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, "relation": data.get("relation")}
            for u, v, data in G.edges(data=True)
        ]
    }
    with open(GRAPH_JSON, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, indent=2, default=str)

    return G


def main():
    df = pd.read_csv(CSV_PATH)
    schema = load_schema()
    graph = create_graph_db(df, schema)

    print(f"Graph schema loaded from: {SCHEMA_PATH}")
    print(f"Knowledge graph saved to: {GRAPH_DB}")
    print(f"JSON export saved to: {GRAPH_JSON}")
    print(f"GraphML export saved to: {GRAPH_GRAPHML}")
    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()
