from pathlib import Path
import re
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path.cwd()
CSV_PATH = BASE_DIR / "outputs" / "MSME_preprocessed_full.csv"
DB_PATH = BASE_DIR / "outputs" / "msme_dashboard.db"

st.set_page_config(page_title="MSME Analytics Dashboard", layout="wide")


def ensure_database():
    if not DB_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("msme", conn, if_exists="replace", index=False)
        conn.close()


def get_connection():
    ensure_database()
    return sqlite3.connect(DB_PATH)


def fetch_data(filters: dict):
    conn = get_connection()
    query = "SELECT * FROM msme WHERE 1=1"
    params = []

    if filters.get("district"):
        query += " AND DISTRICT_NAME = ?"
        params.append(filters["district"])
    if filters.get("category"):
        query += " AND CATEGOERY_NAME = ?"
        params.append(filters["category"])
    if filters.get("industrial_park"):
        query += " AND industrial_park = ?"
        params.append(filters["industrial_park"])
    if filters.get("risk"):
        query += " AND PCBCATEGORY = ?"
        params.append(filters["risk"])
    if filters.get("search"):
        query += " AND (UNIT_NAME LIKE ? OR DISTRICT_NAME LIKE ? OR mandal_name LIKE ?)"
        search_term = f"%{filters['search']}%"
        params.extend([search_term, search_term, search_term])

    query += " LIMIT 1000"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def translate_to_sql(question: str):
    q = question.strip().lower()

    # Detect explicit filters
    if re.search(r"(show|find|list|get).*(women|women-led|female)", q):
        return "SELECT * FROM msme WHERE is_women_led = 1 OR promoter_women = 'Yes' LIMIT 1000;"

    if re.search(r"(high risk|risk|orange|red)", q):
        return "SELECT * FROM msme WHERE PCBCATEGORY IN ('ORANGE', 'RED') LIMIT 1000;"

    if re.search(r"(top|highest|largest).*district", q):
        limit_match = re.search(r"\b(\d+)\b", q)
        limit = int(limit_match.group(1)) if limit_match else 10
        return (
            f"SELECT DISTRICT_NAME, COUNT(*) AS unit_count FROM msme "
            f"GROUP BY DISTRICT_NAME ORDER BY unit_count DESC LIMIT {limit};"
        )

    if re.search(r"(average|avg|mean).*turnover", q):
        return (
            "SELECT DISTRICT_NAME, AVG(turnover) AS avg_turnover "
            "FROM msme WHERE turnover IS NOT NULL GROUP BY DISTRICT_NAME ORDER BY avg_turnover DESC;"
        )

    if re.search(r"(district|mandal).*(category|type)", q):
        return (
            "SELECT DISTRICT_NAME, CATEGOERY_NAME, COUNT(*) AS unit_count "
            "FROM msme GROUP BY DISTRICT_NAME, CATEGOERY_NAME ORDER BY unit_count DESC LIMIT 100;"
        )

    district_match = re.search(r"district\s+['\"]?([A-Za-z ]+)['\"]?", q)
    if district_match:
        district = district_match.group(1).strip().upper()
        return f"SELECT * FROM msme WHERE DISTRICT_NAME = '{district}' LIMIT 1000;"

    category_match = re.search(r"category\s+['\"]?([A-Za-z- ]+)['\"]?", q)
    if category_match:
        category = category_match.group(1).strip().upper()
        return f"SELECT * FROM msme WHERE CATEGOERY_NAME = '{category}' LIMIT 1000;"

    if re.search(r"(graph|visual|show).*", q):
        return "SELECT DISTRICT_NAME, CATEGOERY_NAME, PCBCATEGORY, COUNT(*) AS unit_count FROM msme GROUP BY DISTRICT_NAME, CATEGOERY_NAME, PCBCATEGORY ORDER BY unit_count DESC LIMIT 200;"

    return "SELECT * FROM msme LIMIT 1000;"


def build_knowledge_graph(df: pd.DataFrame):
    flow_df = (
        df.groupby(["DISTRICT_NAME", "CATEGOERY_NAME", "PCBCATEGORY"], dropna=False)
          .size()
          .reset_index(name="count")
    )
    if flow_df.empty:
        return None

    district_nodes = sorted(flow_df["DISTRICT_NAME"].dropna().unique().tolist())
    category_nodes = sorted(flow_df["CATEGOERY_NAME"].dropna().unique().tolist())
    risk_nodes = sorted(flow_df["PCBCATEGORY"].dropna().unique().tolist())

    node_labels = district_nodes + category_nodes + risk_nodes
    node_colors = ["#4C78A8"] * len(district_nodes) + ["#F58518"] * len(category_nodes) + ["#E45756"] * len(risk_nodes)

    district_index = {node: i for i, node in enumerate(district_nodes)}
    category_index = {node: i + len(district_nodes) for i, node in enumerate(category_nodes)}
    risk_index = {node: i + len(district_nodes) + len(category_nodes) for i, node in enumerate(risk_nodes)}

    source = []
    target = []
    value = []

    for _, row in flow_df.iterrows():
        d = row["DISTRICT_NAME"]
        c = row["CATEGOERY_NAME"]
        r = row["PCBCATEGORY"]
        if pd.notna(d) and pd.notna(c) and pd.notna(r):
            source.append(district_index[d])
            target.append(category_index[c])
            value.append(int(row["count"]))
            source.append(category_index[c])
            target.append(risk_index[r])
            value.append(int(row["count"]))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=22,
                    line=dict(color="rgba(0,0,0,0.35)", width=0.8),
                    label=node_labels,
                    color=node_colors,
                    hovertemplate="%{label}<extra></extra>"
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color="rgba(76, 120, 168, 0.3)",
                    hovertemplate="%{source.label} → %{target.label}<br>Count: %{value}<extra></extra>"
                )
            )
        ]
    )
    fig.update_layout(title_text="Knowledge Graph: District → Category → Risk", font=dict(size=12))
    return fig


def main():
    ensure_database()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM msme", conn)
    conn.close()

    st.title("MSME Explorer")
    st.caption("Interactive dashboard for querying and visualizing the MSME dataset")

    with st.sidebar:
        st.header("Filters")
        districts = sorted(df["DISTRICT_NAME"].dropna().unique().tolist())
        categories = sorted(df["CATEGOERY_NAME"].dropna().unique().tolist())
        parks = sorted(df["industrial_park"].dropna().unique().tolist())
        risks = sorted(df["PCBCATEGORY"].dropna().unique().tolist())

        selected_district = st.selectbox("District", ["All"] + districts)
        selected_category = st.selectbox("Category", ["All"] + categories)
        selected_park = st.selectbox("Industrial Park", ["All"] + parks)
        selected_risk = st.selectbox("PCB Category", ["All"] + risks)
        search_term = st.text_input("Search unit / district / mandal")

        st.divider()
        st.subheader("English Query")
        english_query = st.text_area(
            "Ask in English (example: show top 10 districts with high risk units)",
            value="show top 10 districts by unit count",
            height=90,
        )
        run_english_query = st.button("Run English Query")

        st.divider()
        st.subheader("Custom SQL")
        custom_query = st.text_area(
            "Run a SQL query against the local database",
            value="SELECT DISTRICT_NAME, COUNT(*) AS units FROM msme GROUP BY DISTRICT_NAME ORDER BY units DESC LIMIT 10;",
            height=140,
        )
        run_sql_query = st.button("Run SQL Query")

        filters = {
            "district": selected_district if selected_district != "All" else None,
            "category": selected_category if selected_category != "All" else None,
            "industrial_park": selected_park if selected_park != "All" else None,
            "risk": selected_risk if selected_risk != "All" else None,
            "search": search_term.strip() or None,
        }

    filtered = fetch_data(filters)

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    # First display knowledge graph/tab
    graph_tab, results_tab = st.tabs(["Knowledge Graph", "Query Results"])

    with graph_tab:
        graph_fig = build_knowledge_graph(filtered)
        if graph_fig is not None:
            st.plotly_chart(graph_fig, width="stretch")
        else:
            st.info("Not enough data to build the knowledge graph.")

    with results_tab:
        if run_english_query:
            translated_sql = translate_to_sql(english_query)
            st.code(translated_sql, language="sql")
            try:
                conn = get_connection()
                english_result = pd.read_sql_query(translated_sql.rstrip(';'), conn)
                conn.close()
                st.subheader("English Query Result")
                st.dataframe(english_result, width="stretch")
            except Exception as e:
                st.error(f"English query error: {e}")

        if run_sql_query:
            try:
                conn = get_connection()
                custom_result = pd.read_sql_query(custom_query.rstrip(';'), conn)
                conn.close()
                st.subheader("Custom SQL Result")
                st.dataframe(custom_result, width="stretch")
            except Exception as e:
                st.error(f"SQL query error: {e}")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Units", f"{len(filtered):,}")
        col2.metric("Avg Investment", f"₹{filtered['investment'].mean():,.0f}")
        col3.metric("Avg Turnover", f"₹{filtered['turnover'].mean():,.0f}")
        col4.metric("High Risk Units", f"{(filtered['PCBCATEGORY'].isin(['ORANGE', 'RED'])).sum():,}")

        # Query results table
        st.subheader("Filtered Records")
        st.dataframe(
            filtered[[
                "SL", "UNIT_NAME", "DISTRICT_NAME", "mandal_name",
                "CATEGOERY_NAME", "investment", "employment", "turnover",
                "PCBCATEGORY", "risk_score"
            ]].sort_values("risk_score", ascending=False),
            width="stretch"
        )

        # Charts
        colA, colB = st.columns(2)

        with colA:
            district_counts = filtered["DISTRICT_NAME"].value_counts().head(10)
            st.subheader("Top Districts")
            st.plotly_chart(
                px.bar(
                    x=district_counts.index,
                    y=district_counts.values,
                    labels={"x": "District", "y": "Count"},
                    title="District Distribution"
                ),
                width="stretch"
            )

        with colB:
            st.subheader("Risk Distribution")
            risk_counts = filtered["PCBCATEGORY"].value_counts()
            st.plotly_chart(
                px.pie(
                    names=risk_counts.index,
                    values=risk_counts.values,
                    title="PCB Category Share"
                ),
                width="stretch"
            )

        # Geographic scatter
        st.subheader("Geographic Distribution")
        geo = filtered.dropna(subset=["latitude", "longitude"])
        if not geo.empty:
            st.plotly_chart(
                px.scatter_geo(
                    geo,
                    lat="latitude",
                    lon="longitude",
                    color="PCBCATEGORY",
                    hover_name="UNIT_NAME",
                    hover_data=["DISTRICT_NAME", "investment", "turnover"],
                    projection="natural earth",
                    title="MSME Locations"
                ),
                width="stretch"
            )


if __name__ == "__main__":
    main()
