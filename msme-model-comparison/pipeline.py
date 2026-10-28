"""
MSME CI/CD Pipeline — Local
Watches for CSV changes, rebuilds data + relationships + ML model, restarts app.

Usage:
  python pipeline.py              # Run once (rebuild everything)
  python pipeline.py --watch      # Watch mode (auto-rebuild on CSV change)
  python pipeline.py --serve      # Rebuild + start FastAPI + Vite servers
"""
import os, sys, time, subprocess, hashlib, json, argparse

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, 'MSME_new.csv')
DASHBOARD_DIR = os.path.join(BASE, 'msme-dashboard')
ML_DIR = os.path.join(BASE, 'ml_query')
STATE_FILE = os.path.join(BASE, '.pipeline_state.json')

def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def run(cmd, label, cwd=None):
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or BASE, capture_output=True, text=True, encoding='utf-8')
    if result.stdout:
        for line in result.stdout.strip().split('\n')[-15:]:
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:] if result.stderr else 'unknown'}")
        return False
    print(f"  OK")
    return True

def rebuild_all():
    """Full rebuild pipeline."""
    start = time.time()
    print("\n" + "#"*60)
    print("#  MSME Pipeline — Full Rebuild")
    print("#"*60)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        return False

    csv_hash = file_hash(CSV_PATH)
    csv_size = os.path.getsize(CSV_PATH)
    print(f"  CSV: {csv_size/1024/1024:.1f} MB, hash: {csv_hash[:12]}")

    # Step 1: Convert CSV → data.js
    ok = run(
        f'python "{os.path.join(BASE, "convert_data.py")}"',
        "Convert CSV to data.js (all companies)"
    )
    if not ok:
        return False

    # Step 2: Build relationships
    ok = run(
        f'python "{os.path.join(ML_DIR, "find_patterns.py")}"',
        "Build relationships (clustering + link inference)"
    )
    if not ok:
        return False

    # Step 3: Generate ML training data
    ok = run(
        f'python "{os.path.join(ML_DIR, "generate_training_data.py")}"',
        "Generate ML training data"
    )
    if not ok:
        return False

    # Step 4: Retrain TF-IDF model
    ok = run(
        f'''python -c "
import json, pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

base = r'{ML_DIR}'
with open(os.path.join(base, 'train.json')) as f: train = json.load(f)
tq = [d['query'] for d in train]; tl = [d['label'] for d in train]

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3), sublinear_tf=True)
X = tfidf.fit_transform(tq)
cls = {{}}
for f in ['intent','sector','district','category','pcb','sort_by']:
    c = LogisticRegression(max_iter=1000,C=5.0,class_weight='balanced')
    c.fit(X,[l.get(f,'none') for l in tl]); cls[f]=c
for f in ['women','odop','outside_estate','in_estate','has_limit','has_min_investment','has_min_turnover','has_min_employment']:
    c = LogisticRegression(max_iter=1000,C=5.0,class_weight='balanced')
    c.fit(X,[1 if l.get(f.replace('has_','')) or l.get(f) else 0 for l in tl]); cls[f]=c
with open(os.path.join(base,'model_tfidf.pkl'),'wb') as f:
    pickle.dump({{'tfidf':tfidf,'classifiers':cls}},f)
print(f'Trained on {{len(train)}} samples')
"''',
        "Retrain TF-IDF query model"
    )
    if not ok:
        print("  WARNING: ML model retrain failed, continuing with old model")

    # Step 5: Copy JSX to dashboard
    jsx_src = os.path.join(BASE, 'msme_companies.jsx')
    jsx_dst = os.path.join(DASHBOARD_DIR, 'src', 'App.jsx')
    if os.path.exists(jsx_src):
        import shutil
        shutil.copy2(jsx_src, jsx_dst)
        print(f"\n  Copied App.jsx to dashboard")

    # Save state
    elapsed = time.time() - start
    state = load_state()
    state['last_build'] = time.strftime('%Y-%m-%d %H:%M:%S')
    state['csv_hash'] = csv_hash
    state['csv_size'] = csv_size
    state['build_time_sec'] = round(elapsed, 1)
    state['data_js_size'] = os.path.getsize(os.path.join(DASHBOARD_DIR, 'src', 'data.js'))
    state['relationships_size'] = os.path.getsize(os.path.join(DASHBOARD_DIR, 'src', 'relationships.json'))
    save_state(state)

    print(f"\n{'#'*60}")
    print(f"#  Pipeline complete in {elapsed:.1f}s")
    print(f"#  CSV: {csv_size/1024/1024:.1f} MB")
    print(f"#  data.js: {state['data_js_size']/1024/1024:.1f} MB")
    print(f"#  relationships.json: {state['relationships_size']/1024:.0f} KB")
    print(f"{'#'*60}")
    return True

def watch_mode():
    """Watch CSV for changes and auto-rebuild."""
    print("WATCH MODE — monitoring CSV for changes (Ctrl+C to stop)")
    state = load_state()
    last_hash = state.get('csv_hash', '')

    while True:
        try:
            if os.path.exists(CSV_PATH):
                current_hash = file_hash(CSV_PATH)
                if current_hash != last_hash:
                    print(f"\n  CSV changed! ({time.strftime('%H:%M:%S')})")
                    rebuild_all()
                    last_hash = current_hash
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n  Watch stopped.")
            break

def serve_mode():
    """Rebuild then start both servers."""
    rebuild_all()

    print("\n  Starting servers...")

    # Start FastAPI
    api_proc = subprocess.Popen(
        [sys.executable, os.path.join(ML_DIR, 'server.py')],
        cwd=BASE
    )
    print(f"  FastAPI server started (PID {api_proc.pid}) on http://localhost:8000")

    # Start Vite
    vite_proc = subprocess.Popen(
        ['npx', 'vite'],
        cwd=DASHBOARD_DIR,
        shell=True
    )
    print(f"  Vite dev server started (PID {vite_proc.pid})")
    print(f"\n  Dashboard: http://localhost:5173")
    print(f"  API: http://localhost:8000")
    print(f"  Press Ctrl+C to stop all servers")

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\n  Stopping servers...")
        api_proc.terminate()
        vite_proc.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MSME CI/CD Pipeline')
    parser.add_argument('--watch', action='store_true', help='Watch CSV for changes')
    parser.add_argument('--serve', action='store_true', help='Rebuild + start servers')
    args = parser.parse_args()

    if args.watch:
        watch_mode()
    elif args.serve:
        serve_mode()
    else:
        rebuild_all()
