"""Train 3 models and benchmark them on the same test set."""
import os, json, time, pickle, warnings
os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score

BASE = os.path.dirname(__file__)
with open(os.path.join(BASE, 'train.json')) as f: train_data = json.load(f)
with open(os.path.join(BASE, 'test.json')) as f: test_data = json.load(f)

train_queries = [d['query'] for d in train_data]
train_labels = [d['label'] for d in train_data]
test_queries = [d['query'] for d in test_data]
test_labels = [d['label'] for d in test_data]

# All possible field values
ALL_INTENTS = sorted(set(l['intent'] for l in train_labels + test_labels))
ALL_SECTORS = sorted(set(l.get('sector','none') for l in train_labels + test_labels))
ALL_DISTRICTS = sorted(set(l.get('district','none') for l in train_labels + test_labels))
ALL_CATEGORIES = sorted(set(l.get('category','none') for l in train_labels + test_labels))
ALL_PCBS = sorted(set(l.get('pcb','none') for l in train_labels + test_labels))
ALL_SORTS = sorted(set(l.get('sort_by','inv') for l in train_labels + test_labels))

results = {}

def extract_fields(labels):
    return {
        'intent': [l['intent'] for l in labels],
        'sector': [l.get('sector','none') for l in labels],
        'district': [l.get('district','none') for l in labels],
        'category': [l.get('category','none') for l in labels],
        'pcb': [l.get('pcb','none') for l in labels],
        'women': [1 if l.get('women') else 0 for l in labels],
        'odop': [1 if l.get('odop') else 0 for l in labels],
        'outside_estate': [1 if l.get('outside_estate') else 0 for l in labels],
        'in_estate': [1 if l.get('in_estate') else 0 for l in labels],
        'sort_by': [l.get('sort_by','inv') for l in labels],
        'has_limit': [1 if l.get('limit') else 0 for l in labels],
        'limit': [l.get('limit', 0) for l in labels],
        'has_min_investment': [1 if l.get('min_investment') else 0 for l in labels],
        'has_min_turnover': [1 if l.get('min_turnover') else 0 for l in labels],
        'has_min_employment': [1 if l.get('min_employment') else 0 for l in labels],
    }

train_fields = extract_fields(train_labels)
test_fields = extract_fields(test_labels)

def score_model(name, predictions, test_fields):
    scores = {}
    for field in ['intent', 'sector', 'district', 'category', 'pcb', 'sort_by']:
        if field in predictions:
            scores[field] = accuracy_score(test_fields[field], predictions[field])
    for field in ['women', 'odop', 'outside_estate', 'in_estate', 'has_limit', 'has_min_investment']:
        if field in predictions:
            scores[field] = accuracy_score(test_fields[field], predictions[field])
    avg = np.mean(list(scores.values()))
    scores['average'] = avg
    return scores

# ════════════════════════════════════════════════════════════════════════════════
# MODEL 1: TF-IDF + Logistic Regression
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODEL 1: TF-IDF + Logistic Regression")
print("=" * 60)

t0 = time.time()

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), sublinear_tf=True)
X_train = tfidf.fit_transform(train_queries)
X_test = tfidf.transform(test_queries)

classifiers = {}
predictions_tfidf = {}

# Train one classifier per categorical field
for field in ['intent', 'sector', 'district', 'category', 'pcb', 'sort_by']:
    clf = LogisticRegression(max_iter=1000, C=5.0, class_weight='balanced')
    clf.fit(X_train, train_fields[field])
    pred = clf.predict(X_test)
    classifiers[field] = clf
    predictions_tfidf[field] = pred

# Train binary classifiers
for field in ['women', 'odop', 'outside_estate', 'in_estate', 'has_limit', 'has_min_investment', 'has_min_turnover', 'has_min_employment']:
    clf = LogisticRegression(max_iter=1000, C=5.0, class_weight='balanced')
    clf.fit(X_train, train_fields[field])
    pred = clf.predict(X_test)
    classifiers[field] = clf
    predictions_tfidf[field] = pred

train_time_tfidf = time.time() - t0

# Inference speed
t0 = time.time()
for _ in range(100):
    x = tfidf.transform(["give top 5 women led pharma companies"])
    for field, clf in classifiers.items():
        clf.predict(x)
inference_time_tfidf = (time.time() - t0) / 100 * 1000  # ms

scores_tfidf = score_model("TF-IDF", predictions_tfidf, test_fields)
print(f"Training time: {train_time_tfidf:.1f}s")
print(f"Inference time: {inference_time_tfidf:.1f}ms")
print("Accuracy per field:")
for k, v in scores_tfidf.items():
    print(f"  {k}: {v:.3f}")

# Save model
with open(os.path.join(BASE, 'model_tfidf.pkl'), 'wb') as f:
    pickle.dump({'tfidf': tfidf, 'classifiers': classifiers}, f)
model_size_tfidf = os.path.getsize(os.path.join(BASE, 'model_tfidf.pkl')) / 1024

results['tfidf'] = {
    'train_time': train_time_tfidf,
    'inference_ms': inference_time_tfidf,
    'scores': scores_tfidf,
    'model_size_kb': model_size_tfidf,
}

# ════════════════════════════════════════════════════════════════════════════════
# MODEL 2: Sentence Transformers + Logistic Regression
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 2: Sentence Transformers + Logistic Regression")
print("=" * 60)

t0 = time.time()
from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Encoding train set...")
X_train_st = st_model.encode(train_queries, show_progress_bar=True, batch_size=64)
print("Encoding test set...")
X_test_st = st_model.encode(test_queries, show_progress_bar=True, batch_size=64)

classifiers_st = {}
predictions_st = {}

for field in ['intent', 'sector', 'district', 'category', 'pcb', 'sort_by']:
    clf = LogisticRegression(max_iter=1000, C=5.0, class_weight='balanced')
    clf.fit(X_train_st, train_fields[field])
    pred = clf.predict(X_test_st)
    classifiers_st[field] = clf
    predictions_st[field] = pred

for field in ['women', 'odop', 'outside_estate', 'in_estate', 'has_limit', 'has_min_investment', 'has_min_turnover', 'has_min_employment']:
    clf = LogisticRegression(max_iter=1000, C=5.0, class_weight='balanced')
    clf.fit(X_train_st, train_fields[field])
    pred = clf.predict(X_test_st)
    classifiers_st[field] = clf
    predictions_st[field] = pred

train_time_st = time.time() - t0

# Inference speed
t0 = time.time()
for _ in range(20):
    emb = st_model.encode(["give top 5 women led pharma companies"])
    for field, clf in classifiers_st.items():
        clf.predict(emb)
inference_time_st = (time.time() - t0) / 20 * 1000

scores_st = score_model("SentenceTransformer", predictions_st, test_fields)
print(f"Training time: {train_time_st:.1f}s")
print(f"Inference time: {inference_time_st:.1f}ms")
print("Accuracy per field:")
for k, v in scores_st.items():
    print(f"  {k}: {v:.3f}")

with open(os.path.join(BASE, 'model_st.pkl'), 'wb') as f:
    pickle.dump({'classifiers': classifiers_st}, f)
model_size_st = os.path.getsize(os.path.join(BASE, 'model_st.pkl')) / 1024

results['sentence_transformer'] = {
    'train_time': train_time_st,
    'inference_ms': inference_time_st,
    'scores': scores_st,
    'model_size_kb': model_size_st,
    'note': 'Requires all-MiniLM-L6-v2 model (~80MB) at runtime',
}

# ════════════════════════════════════════════════════════════════════════════════
# MODEL 3: Fine-tuned T5-small (NL → JSON)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 3: Fine-tuned T5-small (NL -> JSON)")
print("=" * 60)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

class QueryDataset(Dataset):
    def __init__(self, queries, labels, tokenizer, max_input=128, max_output=128):
        self.queries = queries
        self.labels = [json.dumps(l, separators=(',',':')) for l in labels]
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        inp = self.tokenizer(f"parse: {self.queries[idx]}", max_length=self.max_input,
                             padding='max_length', truncation=True, return_tensors='pt')
        out = self.tokenizer(self.labels[idx], max_length=self.max_output,
                             padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inp.input_ids.squeeze()
        attention_mask = inp.attention_mask.squeeze()
        labels = out.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

t0 = time.time()
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=True)
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
device = 'cpu'
t5_model.to(device)

train_dataset = QueryDataset(train_queries, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(t5_model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50,
                                            num_training_steps=len(train_loader) * 5)

EPOCHS = 5
print(f"Training T5-small for {EPOCHS} epochs on CPU...")
t5_model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(t5_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.4f}")

train_time_t5 = time.time() - t0

# Evaluate
t5_model.eval()
predictions_t5 = {f: [] for f in ['intent','sector','district','category','pcb','sort_by',
                                    'women','odop','outside_estate','in_estate','has_limit','has_min_investment','has_min_turnover','has_min_employment']}

t_inf_start = time.time()
correct_json = 0
for i, query in enumerate(test_queries):
    inp = tokenizer(f"parse: {query}", return_tensors='pt', max_length=128, truncation=True).to(device)
    with torch.no_grad():
        out = t5_model.generate(**inp, max_length=128, num_beams=2)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    try:
        parsed = json.loads(decoded)
        correct_json += 1
    except:
        parsed = {}

    predictions_t5['intent'].append(parsed.get('intent', 'search'))
    predictions_t5['sector'].append(parsed.get('sector', 'none'))
    predictions_t5['district'].append(parsed.get('district', 'none'))
    predictions_t5['category'].append(parsed.get('category', 'none'))
    predictions_t5['pcb'].append(parsed.get('pcb', 'none'))
    predictions_t5['sort_by'].append(parsed.get('sort_by', 'inv'))
    predictions_t5['women'].append(1 if parsed.get('women') else 0)
    predictions_t5['odop'].append(1 if parsed.get('odop') else 0)
    predictions_t5['outside_estate'].append(1 if parsed.get('outside_estate') else 0)
    predictions_t5['in_estate'].append(1 if parsed.get('in_estate') else 0)
    predictions_t5['has_limit'].append(1 if parsed.get('limit') else 0)
    predictions_t5['has_min_investment'].append(1 if parsed.get('min_investment') else 0)
    predictions_t5['has_min_turnover'].append(1 if parsed.get('min_turnover') else 0)
    predictions_t5['has_min_employment'].append(1 if parsed.get('min_employment') else 0)

inference_time_t5 = (time.time() - t_inf_start) / len(test_queries) * 1000

scores_t5 = score_model("T5-small", predictions_t5, test_fields)
print(f"\nTraining time: {train_time_t5:.1f}s")
print(f"Inference time: {inference_time_t5:.1f}ms")
print(f"Valid JSON output: {correct_json}/{len(test_queries)} ({correct_json/len(test_queries)*100:.1f}%)")
print("Accuracy per field:")
for k, v in scores_t5.items():
    print(f"  {k}: {v:.3f}")

# Save model
t5_save_path = os.path.join(BASE, 'model_t5')
t5_model.save_pretrained(t5_save_path)
tokenizer.save_pretrained(t5_save_path)
t5_size = sum(os.path.getsize(os.path.join(t5_save_path, f)) for f in os.listdir(t5_save_path)) / 1024

results['t5_small'] = {
    'train_time': train_time_t5,
    'inference_ms': inference_time_t5,
    'scores': scores_t5,
    'model_size_kb': t5_size,
    'valid_json_pct': correct_json / len(test_queries) * 100,
}

# ════════════════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"\n{'Model':<25} {'Train(s)':<10} {'Infer(ms)':<10} {'Size(KB)':<10} {'Avg Acc':<10}")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<25} {r['train_time']:<10.1f} {r['inference_ms']:<10.1f} {r['model_size_kb']:<10.0f} {r['scores']['average']:<10.3f}")

print("\nField-level accuracy comparison:")
print(f"\n{'Field':<20} {'TF-IDF':<10} {'SentTrans':<10} {'T5-small':<10}")
print("-" * 50)
all_fields = set(list(results['tfidf']['scores'].keys()) + list(results['sentence_transformer']['scores'].keys()) + list(results['t5_small']['scores'].keys()))
for field in sorted(all_fields):
    if field == 'average': continue
    t1 = results['tfidf']['scores'].get(field, 0)
    t2 = results['sentence_transformer']['scores'].get(field, 0)
    t3 = results['t5_small']['scores'].get(field, 0)
    best = max(t1, t2, t3)
    def mark(v): return f"{v:.3f}*" if v == best and v > 0 else f"{v:.3f}"
    print(f"  {field:<18} {mark(t1):<10} {mark(t2):<10} {mark(t3):<10}")
print(f"  {'AVERAGE':<18} {results['tfidf']['scores']['average']:<10.3f} {results['sentence_transformer']['scores']['average']:<10.3f} {results['t5_small']['scores']['average']:<10.3f}")

with open(os.path.join(BASE, 'benchmark_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {os.path.join(BASE, 'benchmark_results.json')}")
