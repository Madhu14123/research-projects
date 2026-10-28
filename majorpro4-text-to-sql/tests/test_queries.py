from pathlib import Path
import sys
import pytest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from query_engine import execute_query, describe_schema
from generation import build_prompt, generate_sql

@pytest.fixture
def data():
    return pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "demo.csv")

def test_aggregate_and_quoted_identifier(data):
    result = execute_query(data, 'SELECT COUNT(*), AVG("Like") FROM mcdonalds_data WHERE tasty = \'Yes\'')
    assert result.rows == [(4, 2.5)]

def test_cte(data):
    result = execute_query(data, "WITH x AS (SELECT Age FROM mcdonalds_data) SELECT MIN(Age) FROM x")
    assert result.rows == [(24,)]

@pytest.mark.parametrize("query", [
    "DELETE FROM mcdonalds_data", "DROP TABLE mcdonalds_data",
    "PRAGMA table_info(mcdonalds_data)", "ATTACH DATABASE ':memory:' AS other",
    "SELECT * FROM mcdonalds_data; DELETE FROM mcdonalds_data;",
    "SELECT load_extension('untrusted')", "SELECT * FROM sqlite_master",
    "WITH x AS (SELECT 1) DELETE FROM mcdonalds_data",
])
def test_mutations_and_external_access_are_rejected(data, query):
    with pytest.raises(ValueError):execute_query(data, query)
    assert execute_query(data, "SELECT COUNT(*) FROM mcdonalds_data").rows == [(6,)]

def test_row_limit_and_empty_result(data):
    result = execute_query(data, "SELECT Age FROM mcdonalds_data", row_limit=2)
    assert len(result.rows) == 2 and result.truncated
    result = execute_query(data, "SELECT Age FROM mcdonalds_data WHERE Age > 100")
    assert result.columns == ["Age"] and result.rows == []

def test_markdown_fences(data):
    assert execute_query(data, "```sql\nSELECT COUNT(*) FROM mcdonalds_data\n```").rows == [(6,)]

def test_expensive_query_is_interrupted(data):
    with pytest.raises(ValueError, match="interrupted"):
        execute_query(data, "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT SUM(n) FROM x", timeout_seconds=0.01)

def test_schema_and_prompt(data):
    schema=describe_schema(data)
    assert '"Like"' in schema and 'Once a week' in schema
    assert 'mcdonalds_data' in build_prompt("Count respondents",schema,False)

@pytest.mark.parametrize("encoder_decoder,expected", [(False,"SELECT 1"),(True,"PROMPT SELECT 1")])
def test_causal_output_excludes_input_prompt(encoder_decoder,expected):
    import torch
    from types import SimpleNamespace
    class Tokenizer:
        model_max_length=512;pad_token_id=0;eos_token_id=1
        def __call__(self,*args,**kwargs):return {"input_ids":torch.tensor([[10,11]])}
        def decode(self,tokens,**kwargs):return "SELECT 1" if tokens.tolist()==[12,13] else "PROMPT SELECT 1"
    class Model:
        config=SimpleNamespace(is_encoder_decoder=encoder_decoder,max_position_embeddings=512)
        def generate(self,**kwargs):return torch.tensor([[10,11,12,13]])
    assert generate_sql("Count", "Table: mcdonalds_data",Tokenizer(),Model())==expected
