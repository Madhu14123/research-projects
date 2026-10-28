from pathlib import Path
import json
import sqlite3
import warnings
from typing import Dict, Any

import networkx as nx
import pandas as pd
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

BASE_DIR = Path.cwd()
CSV_PATH = BASE_DIR / "outputs" / "MSME_preprocessed_full.csv"
SCHEMA_PATH = BASE_DIR / "knowledge_graph_schema_extensive.json"
GRAPH_DB = BASE_DIR / "outputs" / "msme_knowledge_graph_extensive.db"
GRAPH_JSON = BASE_DIR / "outputs" / "msme_knowledge_graph_extensive.json"


def load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(val):
    if pd.isna(val):
        return None
    return str(val).strip()


def dedupe_id(prefix: str, value: Any):
    if value is None:
        return None
    return f"{prefix}:{normalize(value)}"


def derive_risk_bucket(score):
    if score is None:
        return None
    if score >= 0.7:
        return "HIGH_RISK"
    if score >= 0.4:
        return "MEDIUM_RISK"
    return "LOW_RISK"


def derive_scale_band(investment):
    if investment is None:
        return None
    if investment >= 50_000_000:
        return "LARGE_SCALE"
    if investment >= 10_000_000:
        return "MID_SCALE"
    return "SMALL_SCALE"


def derive_growth_signal(roi_ratio):
    if roi_ratio is None:
        return None
    if roi_ratio >= 0.08:
        return "STRONG_GROWTH"
    if roi_ratio >= 0.03:
        return "MODERATE_GROWTH"
    return "LOW_GROWTH"


def derive_operational_profile(row):
    if row.get("zero_employment_flag") and row.get("zero_turnover_flag"):
        return "NON_OPERATING"
    if row.get("zero_employment_flag"):
        return "EMPLOYMENT_STRESSED"
    if row.get("zero_turnover_flag"):
        return "TURNOVER_STRESSED"
    return "ACTIVE_OPERATION"


def derive_compliance_signal(pcb_category_risk):
    if pcb_category_risk is None:
        return None
    if pcb_category_risk >= 3:
        return "HIGH_COMPLIANCE_RISK"
    if pcb_category_risk == 2:
        return "MODERATE_COMPLIANCE_RISK"
    return "LOW_COMPLIANCE_RISK"


def create_extensive_graph(df: pd.DataFrame, schema: Dict[str, Any]):
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

    G = nx.DiGraph()

    # Add cluster nodes using geospatial similarity
    geo_df = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(geo_df) >= 3:
        k = min(8, len(geo_df))
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        geo_df["cluster_id"] = km.fit_predict(geo_df[["latitude", "longitude"]])
        for cluster_id in sorted(geo_df["cluster_id"].unique()):
            cluster_rows = geo_df[geo_df["cluster_id"] == cluster_id]
            centroid_lat = cluster_rows["latitude"].mean()
            centroid_lon = cluster_rows["longitude"].mean()
            cid = dedupe_id("GeospatialCluster", f"cluster_{int(cluster_id)}")
            if cid:
                G.add_node(
                    cid,
                    node_type="GeospatialCluster",
                    label=f"cluster_{int(cluster_id)}",
                    cluster_id=int(cluster_id),
                    cluster_center_lat=float(centroid_lat),
                    cluster_center_lon=float(centroid_lon)
                )

    # Add nodes and edges for each MSME row
    for _, row in df.iterrows():
        m_id = dedupe_id("MSME", row.get("SL", row.get("UNIT_NAME")))
        if not m_id:
            continue

        label = normalize(row.get("UNIT_NAME")) or f"MSME-{normalize(row.get('SL'))}"
        G.add_node(
            m_id,
            node_type="MSME",
            label=label,
            SL=normalize(row.get("SL")),
            UNIT_NAME=normalize(row.get("UNIT_NAME")),
            investment=row.get("investment"),
            employment=row.get("employment"),
            turnover=row.get("turnover"),
            latitude=row.get("latitude"),
            longitude=row.get("longitude"),
            risk_score=row.get("risk_score"),
            ROI_ratio=row.get("ROI_ratio"),
            revenue_per_employee=row.get("revenue_per_employee"),
            employees_per_lakh_investment=row.get("employees_per_lakh_investment"),
            female_percentage=row.get("female_percentage"),
            zero_employment_flag=bool(row.get("zero_employment_flag", 0)),
            zero_turnover_flag=bool(row.get("zero_turnover_flag", 0)),
            is_women_led=bool(row.get("is_women_led", 0)),
            is_odop_participant=bool(row.get("is_odop_participant", 0)),
            is_in_industrial_estate=bool(row.get("is_in_industrial_estate", 0)),
            turnover_missing_flag=bool(row.get("turnover_missing_flag", 0)),
            outlier_flag=bool(row.get("outlier_flag", 0)),
            pcb_category_risk=row.get("pcb_category_risk")
        )

        # Derived risk bucket
        risk_bucket = derive_risk_bucket(row.get("risk_score"))
        if risk_bucket:
            rb_id = dedupe_id("RiskBucket", risk_bucket)
            if rb_id:
                G.add_node(rb_id, node_type="RiskBucket", label=risk_bucket, risk_bucket=risk_bucket)
                G.add_edge(m_id, rb_id, relation="HAS_RISK_BUCKET")

        # Derived financial profile
        financial_profile_name = None
        if row.get("investment") and row.get("turnover"):
            if row.get("ROI_ratio") is not None:
                if row.get("ROI_ratio") >= 5:
                    financial_profile_name = "HIGH_ROI"
                elif row.get("ROI_ratio") >= 2:
                    financial_profile_name = "MID_ROI"
                else:
                    financial_profile_name = "LOW_ROI"
        if financial_profile_name:
            fp_id = dedupe_id("FinancialProfile", financial_profile_name)
            if fp_id:
                G.add_node(fp_id, node_type="FinancialProfile", label=financial_profile_name, profile_name=financial_profile_name)
                G.add_edge(m_id, fp_id, relation="HAS_FINANCIAL_PROFILE")

        # Additional derived profiles for richer graph semantics
        scale_band = derive_scale_band(row.get("investment"))
        if scale_band:
            sb_id = dedupe_id("ScaleBand", scale_band)
            if sb_id:
                G.add_node(sb_id, node_type="ScaleBand", label=scale_band, band_name=scale_band)
                G.add_edge(m_id, sb_id, relation="HAS_SCALE_BAND")

        growth_signal = derive_growth_signal(row.get("ROI_ratio"))
        if growth_signal:
            gs_id = dedupe_id("GrowthSignal", growth_signal)
            if gs_id:
                G.add_node(gs_id, node_type="GrowthSignal", label=growth_signal, signal_name=growth_signal)
                G.add_edge(m_id, gs_id, relation="HAS_GROWTH_SIGNAL")

        operational_profile = derive_operational_profile(row)
        op_id = dedupe_id("OperationalProfile", operational_profile)
        if op_id:
            G.add_node(op_id, node_type="OperationalProfile", label=operational_profile, profile_name=operational_profile)
            G.add_edge(m_id, op_id, relation="HAS_OPERATIONAL_PROFILE")

        compliance_signal = derive_compliance_signal(row.get("pcb_category_risk"))
        if compliance_signal:
            cs_id = dedupe_id("ComplianceSignal", compliance_signal)
            if cs_id:
                G.add_node(cs_id, node_type="ComplianceSignal", label=compliance_signal, signal_name=compliance_signal)
                G.add_edge(m_id, cs_id, relation="HAS_COMPLIANCE_SIGNAL")

        ownership_pattern = "WOMEN_LED" if bool(row.get("is_women_led", 0)) else "GENERAL_OWNERSHIP"
        ow_id = dedupe_id("OwnershipPattern", ownership_pattern)
        if ow_id:
            G.add_node(ow_id, node_type="OwnershipPattern", label=ownership_pattern, pattern_name=ownership_pattern)
            G.add_edge(m_id, ow_id, relation="HAS_OWNERSHIP_PATTERN")

        # Simple cluster association
        if "cluster_id" in geo_df.columns and m_id in G.nodes:
            row_cluster = geo_df.loc[geo_df["SL"] == row.get("SL"), "cluster_id"]
            if not row_cluster.empty:
                cluster_id = int(row_cluster.iloc[0])
                cid = dedupe_id("GeospatialCluster", f"cluster_{cluster_id}")
                if cid and cid in G.nodes:
                    G.add_edge(m_id, cid, relation="BELONGS_TO_CLUSTER")

        # Entity nodes and edges (standard schema)
        entity_edges = [
            ("DISTRICT_NAME", "District", "LOCATED_IN_DISTRICT"),
            ("mandal_name", "Mandal", "LOCATED_IN_MANDAL"),
            ("CATEGOERY_NAME", "Category", "BELONGS_TO_CATEGORY"),
            ("industrial_park", "IndustrialPark", "HAS_INDUSTRIAL_PARK"),
            ("PCBCATEGORY", "PCBCategory", "HAS_PCB_CATEGORY"),
            ("social_status", "SocialStatus", "HAS_SOCIAL_STATUS"),
            ("promoter_women", "PromoterType", "HAS_PROMOTER_TYPE"),
            ("ODOP", "ODOPFlag", "PARTICIPATES_IN_ODOP")
        ]

        for col, node_type, relation in entity_edges:
            value = normalize(row.get(col))
            if value is None:
                continue
            node_id = dedupe_id(node_type, value)
            if not node_id:
                continue
            G.add_node(node_id, node_type=node_type, label=value)
            G.add_edge(m_id, node_id, relation=relation)

    # Add district-level aggregation edges
    district_df = df.groupby("DISTRICT_NAME", dropna=False).agg(
        district_enterprise_count=("SL", "count")
    ).reset_index()
    for _, row in district_df.iterrows():
        district_name = normalize(row.get("DISTRICT_NAME"))
        if district_name is None:
            continue
        district_id = dedupe_id("District", district_name)
        if district_id in G.nodes:
            G.nodes[district_id]["district_enterprise_count"] = int(row.get("district_enterprise_count", 0))
            # add category relationships by district
            district_categories = df[df["DISTRICT_NAME"] == district_name].groupby("CATEGOERY_NAME", dropna=False).size()
            for category_name, count in district_categories.items():
                category_id = dedupe_id("Category", normalize(category_name))
                if category_id in G.nodes:
                    G.add_edge(district_id, category_id, relation="DISTRICT_HAS_CATEGORY", count=int(count))

    # Save nodes and edges in sqlite
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

    graph_json = {
        "nodes": [
            {"id": node_id, "type": data.get("node_type"), "label": data.get("label"), "properties": data}
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
    graph = create_extensive_graph(df, schema)
    print(f"Extensive schema loaded from: {SCHEMA_PATH}")
    print(f"Graph DB saved to: {GRAPH_DB}")
    print(f"JSON export saved to: {GRAPH_JSON}")
    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()
