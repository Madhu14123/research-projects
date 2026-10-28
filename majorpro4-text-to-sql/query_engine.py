"""Schema-aware, bounded read-only SQLite queries over a CSV table."""
from dataclasses import dataclass
import re
import sqlite3
import time

TABLE_NAME = "mcdonalds_data"

@dataclass
class QueryResult:
    columns: list
    rows: list
    truncated: bool

def clean_sql(text):
    text = text.strip()
    fenced = re.fullmatch(r"```(?:sql|sqlite)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text

def describe_schema(frame):
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The CSV must contain columns and at least one row.")
    lines = [f"Table: {TABLE_NAME}"]
    for column in frame.columns:
        values = frame[column].dropna()
        examples = list(dict.fromkeys(str(v) for v in values))[:5]
        quoted = '"' + str(column).replace('"', '""') + '"'
        lines.append(f"{quoted} ({frame[column].dtype}); examples: {examples!r}")
    return "\n".join(lines)

_READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
_FUNCTIONS = set("count avg sum total min max round abs coalesce ifnull nullif lower upper length trim ltrim rtrim substr substring replace like glob printf format date time datetime julianday strftime unixepoch row_number rank dense_rank percent_rank cume_dist ntile lag lead first_value last_value nth_value".split())

def execute_query(frame, sql, row_limit=1000, timeout_seconds=3.0):
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("Load a CSV with at least one row first.")
    if not 1 <= row_limit <= 10000 or not 0 < timeout_seconds <= 30:
        raise ValueError("Invalid query limits.")
    sql = clean_sql(sql)
    if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Enter one SELECT query, optionally beginning with WITH.")
    connection = sqlite3.connect(":memory:")
    try:
        frame.to_sql(TABLE_NAME, connection, index=False, if_exists="replace")
        connection.execute("PRAGMA query_only = ON")
        def authorizer(action, arg1, arg2, database, source):
            if action not in _READ_ACTIONS:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and arg1 != TABLE_NAME:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() not in _FUNCTIONS:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        connection.set_authorizer(authorizer)
        deadline = time.monotonic() + timeout_seconds
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(row_limit + 1)
        return QueryResult([d[0] for d in cursor.description], rows[:row_limit], len(rows) > row_limit)
    except sqlite3.Error as exc:
        raise ValueError(f"Query could not run: {exc}") from exc
    finally:
        connection.close()
