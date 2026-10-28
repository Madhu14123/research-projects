"""Train T5-small model only and benchmark."""
import os, json, time, sys
os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(__file__)
with open(os.path.join(BASE, 'train.json')) as f: train_data = json.load(f)
with open(os.path.join(BASE, 'test.json')) as f: test_data = json.load(f)

train_queries = [d['query'] for d in train_data]
train_labels = [d['label'] for d in train_data]
test_queries = [d['query'] for d in test_data]
test_labels = [d['label'] for d in test_data]

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

print("=" * 60)
print("MODEL 3: Fine-tuned T5-small (NL -> JSON)")
print("=" * 60)

t0 = time.time()
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=True)
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
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

    print(f"  Epoch {epoch+1}/{EPOCHS} -- Loss: {total_loss/len(train_loader):.4f}")

train_time = time.time() - t0

# Evaluate
t5_model.eval()
correct_json = 0
field_preds = {f: [] for f in ['intent','sector','district','category','pcb','sort_by',
                                'women','odop','outside_estate','in_estate','has_limit','has_min_investment']}
field_true = {
    'intent': [l['intent'] for l in test_labels],
    'sector': [l.get('sector','none') for l in test_labels],
    'district': [l.get('district','none') for l in test_labels],
    'category': [l.get('category','none') for l in test_labels],
    'pcb': [l.get('pcb','none') for l in test_labels],
    'sort_by': [l.get('sort_by','inv') for l in test_labels],
    'women': [1 if l.get('women') else 0 for l in test_labels],
    'odop': [1 if l.get('odop') else 0 for l in test_labels],
    'outside_estate': [1 if l.get('outside_estate') else 0 for l in test_labels],
    'in_estate': [1 if l.get('in_estate') else 0 for l in test_labels],
    'has_limit': [1 if l.get('limit') else 0 for l in test_labels],
    'has_min_investment': [1 if l.get('min_investment') else 0 for l in test_labels],
}

t_inf = time.time()
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

    field_preds['intent'].append(parsed.get('intent', 'search'))
    field_preds['sector'].append(parsed.get('sector', 'none'))
    field_preds['district'].append(parsed.get('district', 'none'))
    field_preds['category'].append(parsed.get('category', 'none'))
    field_preds['pcb'].append(parsed.get('pcb', 'none'))
    field_preds['sort_by'].append(parsed.get('sort_by', 'inv'))
    field_preds['women'].append(1 if parsed.get('women') else 0)
    field_preds['odop'].append(1 if parsed.get('odop') else 0)
    field_preds['outside_estate'].append(1 if parsed.get('outside_estate') else 0)
    field_preds['in_estate'].append(1 if parsed.get('in_estate') else 0)
    field_preds['has_limit'].append(1 if parsed.get('limit') else 0)
    field_preds['has_min_investment'].append(1 if parsed.get('min_investment') else 0)

inference_ms = (time.time() - t_inf) / len(test_queries) * 1000

print(f"\nTraining time: {train_time:.1f}s")
print(f"Inference time: {inference_ms:.1f}ms per query")
print(f"Valid JSON output: {correct_json}/{len(test_queries)} ({correct_json/len(test_queries)*100:.1f}%)")
print("\nAccuracy per field:")
scores = {}
for field in field_preds:
    acc = accuracy_score(field_true[field], field_preds[field])
    scores[field] = acc
    print(f"  {field}: {acc:.3f}")
avg = np.mean(list(scores.values()))
print(f"  AVERAGE: {avg:.3f}")

# Save
t5_save_path = os.path.join(BASE, 'model_t5')
t5_model.save_pretrained(t5_save_path)
tokenizer.save_pretrained(t5_save_path)
t5_size = sum(os.path.getsize(os.path.join(t5_save_path, f)) for f in os.listdir(t5_save_path)) / 1024
print(f"\nModel size: {t5_size:.0f} KB")

# Final comparison with saved results
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"{'Model':<25} {'Train(s)':<10} {'Infer(ms)':<12} {'Avg Acc':<10}")
print("-" * 57)
print(f"{'TF-IDF + LogReg':<25} {'0.7':<10} {'2.2':<12} {'0.986':<10}")
print(f"{'SentenceTransformer':<25} {'29.6':<10} {'16.2':<12} {'0.984':<10}")
print(f"{'T5-small fine-tuned':<25} {train_time:<10.1f} {inference_ms:<12.1f} {avg:<10.3f}")
print(f"\nT5 valid JSON rate: {correct_json/len(test_queries)*100:.1f}%")
