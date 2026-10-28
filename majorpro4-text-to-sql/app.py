"""CSV question answering with reviewable SQL and local model inference."""
from pathlib import Path
import pandas as pd
import streamlit as st
from generation import MODELS, load_model, generate_sql
from query_engine import describe_schema, execute_query

st.set_page_config(page_title="CSV Questions to SQL", layout="wide")
st.title("Ask questions about a CSV")
st.write("Load a CSV, generate a query, review it, and explore the results.")

@st.cache_resource(max_entries=1)
def cached_model(model_id):
    return load_model(model_id)

uploaded = st.file_uploader("Choose a CSV", type=["csv"])
try:
    if uploaded is not None:
        if uploaded.size > 10 * 1024 * 1024:
            raise ValueError("Choose a CSV smaller than 10 MB.")
        frame = pd.read_csv(uploaded)
    else:
        frame = pd.read_csv(Path(__file__).parent / "data" / "demo.csv")
        st.caption("Showing synthetic demonstration data. Upload your own CSV to explore another dataset.")
    schema = describe_schema(frame)
except Exception as exc:
    st.error(f"Could not load CSV: {exc}")
    st.stop()

with st.expander("Preview data and schema"):
    st.dataframe(frame.head(10), hide_index=True)
    st.code(schema, language="text")

question = st.text_input("Question", placeholder="How many respondents find the food tasty?")
with st.expander("Model settings"):
    selected = st.selectbox("Local model", list(MODELS))
    st.caption("Models download only when you choose Generate SQL. Llama 2 requires model access, HF_TOKEN and substantial memory. Generated SQL can be incorrect; review it before running.")

if "sql_editor" not in st.session_state:
    st.session_state.sql_editor = 'SELECT COUNT(*) AS respondents FROM mcdonalds_data WHERE "tasty" = \'Yes\';'
if st.button("Generate SQL"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Generating a query..."):
                tokenizer, model = cached_model(MODELS[selected])
                st.session_state.sql_editor = generate_sql(question, schema, tokenizer, model)
        except Exception as exc:
            st.error(f"Could not generate SQL: {exc}")

sql = st.text_area("Review or edit SQL", key="sql_editor", height=160)
if st.button("Run SQL"):
    try:
        result = execute_query(frame, sql)
        st.dataframe(pd.DataFrame(result.rows, columns=result.columns), hide_index=True)
        if result.truncated:
            st.caption("Showing the first 1,000 result rows.")
        elif not result.rows:
            st.info("The query ran successfully and returned no rows.")
    except ValueError as exc:
        st.error(str(exc))
