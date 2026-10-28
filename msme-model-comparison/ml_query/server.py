"""FastAPI server for ML-powered MSME query parsing. Uses TF-IDF + LogReg model."""
import os, pickle, re, json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, 'model_tfidf.pkl'), 'rb') as f:
    model = pickle.load(f)

tfidf = model['tfidf']
classifiers = model['classifiers']

app = FastAPI(title="MSME Query Parser")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    intent: str
    sector: str | None = None
    district: str | None = None
    category: str | None = None
    categories: list[str] | None = None
    pcb: str | None = None
    women: bool = False
    odop: bool = False
    outside_estate: bool = False
    in_estate: bool = False
    limit: int | None = None
    sort_by: str = "inv"
    min_investment: int | None = None
    min_turnover: int | None = None
    min_employment: int | None = None
    target_company: str | None = None
    exclude_district: str | None = None
    exclude_category: str | None = None

def extract_number(query, patterns):
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None

@app.post("/parse", response_model=QueryResponse)
def parse_query(req: QueryRequest):
    q = req.query.strip()
    q_lower = q.lower()
    X = tfidf.transform([q])

    result = {}

    # ML predictions
    for field in ['intent', 'sector', 'district', 'category', 'pcb', 'sort_by']:
        pred = classifiers[field].predict(X)[0]
        if pred != 'none':
            result[field] = pred

    for field in ['women', 'odop', 'outside_estate', 'in_estate', 'has_limit', 'has_min_investment', 'has_min_turnover', 'has_min_employment']:
        pred = classifiers[field].predict(X)[0]
        if pred == 1:
            result[field] = True

    # Rule-based extraction for numeric values (ML can't predict exact numbers)
    # Limit
    lim = extract_number(q_lower, [
        r'(?:top|only|first|show|give|just|limit)\s+(\d+)',
        r'(\d+)\s+(?:companies|results|units|firms)',
    ])
    if lim:
        result['limit'] = min(lim, 200)
    elif result.pop('has_limit', None):
        result['limit'] = 10

    # Investment
    inv = extract_number(q_lower, [r'(\d+)\s*(?:crore|cr\b)'])
    if inv:
        result['min_investment'] = inv * 100
    else:
        inv = extract_number(q_lower, [r'(\d+)\s*(?:lakh|lakhs|l\b)'])
        if inv:
            result['min_investment'] = inv
    if not result.get('min_investment'):
        result.pop('has_min_investment', None)
    else:
        result.pop('has_min_investment', None)

    # Turnover
    turn = extract_number(q_lower, [r'turnover.*?(\d+)\s*(?:crore|cr\b)'])
    if turn:
        result['min_turnover'] = turn * 100
    else:
        turn = extract_number(q_lower, [r'turnover.*?(\d+)\s*(?:lakh|lakhs|l\b)'])
        if turn:
            result['min_turnover'] = turn
    result.pop('has_min_turnover', None)

    # Employment
    emp = extract_number(q_lower, [r'(\d+)\+?\s*(?:employ|worker|people|staff|emp\b)'])
    if emp:
        result['min_employment'] = emp
    result.pop('has_min_employment', None)

    # Multi-category
    if re.search(r'large.{1,5}medium|medium.{1,5}large', q_lower):
        result['categories'] = ["LARGE", "MEDIUM"]
        result.pop('category', None)
    if re.search(r'small.{1,5}micro|micro.{1,5}small', q_lower):
        result['categories'] = ["SMALL", "MICRO"]
        result.pop('category', None)

    # Target company
    big_names = ["amara raja","dr. reddy","dr.reddy","aurobindo","kitex","ramagundam",
                 "hetero","natco","granules","laurus","msn lab","sagar cement",
                 "orient cement","mahindra","mrf","hpcl","itc limited","welspun",
                 "bharath electron","sanofi","unique biotech"]
    for name in big_names:
        if name in q_lower:
            result['target_company'] = name
            break

    # Exclusions
    excl = re.search(r'(?:not in|excluding|except)\s+([a-zA-Z]+)', q_lower)
    if excl:
        result['exclude_district'] = excl.group(1).title()
    if re.search(r'excluding\s+micro|not\s+micro|no\s+micro', q_lower):
        result['exclude_category'] = "MICRO"

    # Clean up internal flags
    for k in ['has_limit', 'has_min_investment', 'has_min_turnover', 'has_min_employment']:
        result.pop(k, None)

    return QueryResponse(**result)

@app.get("/health")
def health():
    return {"status": "ok", "model": "tfidf_logreg", "accuracy": 0.986}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
