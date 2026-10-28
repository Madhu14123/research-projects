"""
Find real patterns in MSME data using ONLY the actual fields — no sector guessing.
Uses clustering on: investment, employment, turnover, PCB tier, estate status, size tier, geography.
"""
import csv, sys, json, os
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.stdout.reconfigure(encoding='utf-8')

CATS_MAP = {'MICRO':0,'Micro':0,'SMALL':1,'Small':1,'MEDIUM':2,'Medium':2,'LARGE':3,'Large':3,'MEGA':4,'Mega':4}
PCBS_MAP = {'GREEN':0,'Green':0,'ORANGE':1,'Orange':1,'RED':2,'Red':2,'WHITE':3,'White':3}
CATS = ['MICRO','SMALL','MEDIUM','LARGE','MEGA']

rows = []
with open(os.path.join(os.path.dirname(__file__), '..', 'MSME_new.csv'), 'r', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        try:
            name = (r.get('UNIT_NAME') or '').strip()
            dist = (r.get('DISTRICT_NAME') or '').strip()
            mandal = (r.get('mandal_name') or '').strip()
            cat = CATS_MAP.get((r.get('CATEGOERY_NAME') or '').strip(), 0)
            pcb = PCBS_MAP.get((r.get('PCBCATEGORY') or '').strip(), 3)
            inv = float(r['investment']) if r.get('investment') else 0
            emp = int(float(r['employment'])) if r.get('employment') else 0
            turn = float(r['turnover']) if r.get('turnover') else 0
            park = (r.get('industrial_park') or '')
            estate = 1 if 'In Industrial' in park and 'Not' not in park else 0
            women = 1 if (r.get('promoter_women') or '').strip().upper() == 'YES' else 0
            odop = 1 if (r.get('ODOP') or '').strip().upper() == 'YES' else 0
            lat = float(r['latitude']) if r.get('latitude') else 0
            lon = float(r['longitude']) if r.get('longitude') else 0
            sl = int(r['SL'])
            if dist:
                rows.append({
                    'id':sl,'name':name,'dist':dist,'mandal':mandal,
                    'cat':cat,'pcb':pcb,'inv':inv,'emp':emp,'turn':turn,
                    'estate':estate,'women':women,'odop':odop,'lat':lat,'lon':lon
                })
        except:
            pass

print(f"Loaded {len(rows)} companies\n")

# ════════════════════════════════════════════════════════════
# PATTERN 1: Mandal clusters — natural industrial zones
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("PATTERN 1: MANDAL INDUSTRIAL CLUSTERS")
print("=" * 60)

mandal_groups = defaultdict(list)
for r in rows:
    if r['mandal']:
        mandal_groups[(r['dist'], r['mandal'])].append(r)

# A meaningful cluster: 10+ companies in same mandal
clusters = []
for (dist, mandal), companies in mandal_groups.items():
    if len(companies) >= 10:
        inv_total = sum(c['inv'] for c in companies)
        emp_total = sum(c['emp'] for c in companies)
        cats = Counter(c['cat'] for c in companies)
        pcbs = Counter(c['pcb'] for c in companies)
        estate_pct = sum(c['estate'] for c in companies) / len(companies)

        # Size diversity score: how many different tiers present
        tier_diversity = len([c for c in cats.values() if c > 0])

        # Has anchor? (LARGE or MEGA present)
        has_anchor = cats.get(3, 0) + cats.get(4, 0)

        # Supply chain potential: anchor + small companies = buyer-supplier ecosystem
        small_count = cats.get(0, 0) + cats.get(1, 0)

        clusters.append({
            'dist': dist, 'mandal': mandal,
            'count': len(companies), 'inv': inv_total, 'emp': emp_total,
            'cats': dict(cats), 'pcbs': dict(pcbs),
            'estate_pct': round(estate_pct, 2),
            'tier_diversity': tier_diversity,
            'has_anchor': has_anchor,
            'small_count': small_count,
            'top_companies': sorted(companies, key=lambda x: -x['inv'])[:5],
            'companies': companies,
        })

clusters.sort(key=lambda x: -x['count'])
print(f"Found {len(clusters)} mandal clusters (10+ companies)")
print(f"\nTop 15 clusters:")
for c in clusters[:15]:
    tiers = '+'.join(f"{CATS[k][0]}{v}" for k,v in sorted(c['cats'].items()))
    anchor = "ANCHOR" if c['has_anchor'] > 0 else "no anchor"
    print(f"  {c['dist']:15s}/{c['mandal']:15s}: {c['count']:4d} co. inv={c['inv']/100:.0f}L [{tiers}] {anchor} estate={c['estate_pct']:.0%}")

# ════════════════════════════════════════════════════════════
# PATTERN 2: Economic profile clustering (K-means)
# ════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("PATTERN 2: ECONOMIC PROFILE CLUSTERS (K-means)")
print("=" * 60)

# Features: log(investment), log(employment+1), log(turnover+1), cat, pcb, estate
features = []
for r in rows:
    features.append([
        np.log1p(r['inv']),
        np.log1p(r['emp']),
        np.log1p(r['turn']),
        r['cat'],
        r['pcb'],
        r['estate'],
    ])
X = np.array(features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k
scores = {}
for k in [4, 5, 6, 7, 8]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
    scores[k] = score
    print(f"  k={k}: silhouette={score:.3f}")

best_k = max(scores, key=scores.get)
print(f"\nBest k={best_k} (silhouette={scores[best_k]:.3f})")

km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

# Analyze clusters
print(f"\nCluster profiles:")
for ci in range(best_k):
    mask = labels == ci
    cl_rows = [rows[i] for i in range(len(rows)) if mask[i]]
    avg_inv = np.mean([r['inv'] for r in cl_rows])
    avg_emp = np.mean([r['emp'] for r in cl_rows])
    avg_turn = np.mean([r['turn'] for r in cl_rows])
    cats = Counter(r['cat'] for r in cl_rows)
    pcbs = Counter(r['pcb'] for r in cl_rows)
    estate_pct = sum(r['estate'] for r in cl_rows) / len(cl_rows)
    dominant_cat = CATS[cats.most_common(1)[0][0]]
    dominant_pcb = ['GREEN','ORANGE','RED','WHITE'][pcbs.most_common(1)[0][0]]

    # Name the cluster by its profile
    if avg_inv > 1000 and avg_emp > 100:
        profile = "Large industrial"
    elif avg_inv > 100 and estate_pct > 0.3:
        profile = "Estate-based mid"
    elif avg_emp > 20 and avg_inv < 100:
        profile = "Labor-intensive"
    elif avg_inv < 30 and avg_emp < 10:
        profile = "Micro home-based"
    else:
        profile = "Mixed small"

    print(f"  Cluster {ci} [{profile}]: n={len(cl_rows):,} avg_inv={avg_inv:.0f}L avg_emp={avg_emp:.0f} "
          f"cat={dominant_cat} pcb={dominant_pcb} estate={estate_pct:.0%}")

# Assign cluster labels to rows
for i, r in enumerate(rows):
    r['econ_cluster'] = int(labels[i])

# ════════════════════════════════════════════════════════════
# PATTERN 3: LINK INFERENCE — based on actual data patterns
# ════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("PATTERN 3: LINK INFERENCE (data-driven)")
print("=" * 60)

links = []
link_reasons = []

for (dist, mandal), companies in mandal_groups.items():
    if len(companies) < 5:
        continue

    # Sort by size tier then investment
    companies.sort(key=lambda x: (-x['cat'], -x['inv']))

    # LINK TYPE 1: Size hierarchy in same mandal
    # LARGE/MEGA companies likely buy from SMALL/MICRO in the same mandal
    anchors = [c for c in companies if c['cat'] >= 3]  # LARGE, MEGA
    suppliers = [c for c in companies if c['cat'] <= 1]  # MICRO, SMALL

    for anchor in anchors[:3]:  # top 3 anchors
        # Find suppliers in reasonable investment range (not too tiny, not too big)
        viable = [s for s in suppliers if s['inv'] < anchor['inv'] * 0.5 and s['inv'] > anchor['inv'] * 0.001]
        if not viable:
            viable = suppliers[:10]

        for supplier in viable[:5]:
            confidence = 0.5  # base: same mandal + size hierarchy

            # Boost if both in estate (shared infrastructure)
            if anchor['estate'] and supplier['estate']:
                confidence += 0.15

            # Boost if same economic cluster
            if anchor.get('econ_cluster') == supplier.get('econ_cluster'):
                confidence += 0.1

            # Boost if supplier employs enough people (not a shell company)
            if supplier['emp'] >= 5:
                confidence += 0.1

            # Reduce if different PCB tiers (regulatory friction)
            if abs(anchor['pcb'] - supplier['pcb']) >= 2:
                confidence -= 0.1

            confidence = round(min(max(confidence, 0.2), 1.0), 2)

            reason_parts = [f"Same mandal ({mandal}, {dist})"]
            reason_parts.append(f"Size hierarchy: {CATS[supplier['cat']]} supplies to {CATS[anchor['cat']]}")
            if anchor['estate'] and supplier['estate']:
                reason_parts.append("Both in industrial estate (shared infrastructure)")
            if supplier['emp'] >= 5:
                reason_parts.append(f"Supplier has {supplier['emp']} employees (operational)")

            links.append({
                'source': supplier['id'],
                'target': anchor['id'],
                'type': 'size_hierarchy',
                'confidence': confidence,
                'district': dist,
                'mandal': mandal,
            })
            link_reasons.append(reason_parts)

    # LINK TYPE 2: Estate co-location
    # Companies in the same industrial estate were placed there to form supply chains
    estate_cos = [c for c in companies if c['estate']]
    if len(estate_cos) >= 3:
        estate_anchors = [c for c in estate_cos if c['cat'] >= 2]
        estate_small = [c for c in estate_cos if c['cat'] <= 1]
        for anchor in estate_anchors[:3]:
            for supplier in estate_small[:3]:
                if anchor['id'] == supplier['id']:
                    continue
                links.append({
                    'source': supplier['id'],
                    'target': anchor['id'],
                    'type': 'estate_colocation',
                    'confidence': 0.6,
                    'district': dist,
                    'mandal': mandal,
                })
                link_reasons.append([
                    f"Both in industrial estate in {mandal}, {dist}",
                    "Industrial estates are designed for supply chain clustering",
                    f"Size: {CATS[supplier['cat']]} can supply {CATS[anchor['cat']]}",
                ])

    # LINK TYPE 3: ODOP chain
    # ODOP companies in same district produce the district's specialty product
    odop_cos = [c for c in companies if c['odop']]
    if len(odop_cos) >= 2:
        odop_big = sorted([c for c in odop_cos if c['cat'] >= 1], key=lambda x: -x['inv'])
        odop_small = sorted([c for c in odop_cos if c['cat'] == 0], key=lambda x: -x['inv'])
        for big in odop_big[:2]:
            for small in odop_small[:3]:
                if big['id'] == small['id']:
                    continue
                links.append({
                    'source': small['id'],
                    'target': big['id'],
                    'type': 'odop_chain',
                    'confidence': 0.55,
                    'district': dist,
                    'mandal': mandal,
                })
                link_reasons.append([
                    f"Both ODOP companies in {mandal}, {dist}",
                    "ODOP = One District One Product — both contribute to district's specialty",
                    f"{CATS[small['cat']]} feeds into {CATS[big['cat']]}'s production chain",
                ])

# Deduplicate (keep highest confidence)
seen = set()
deduped = []
for l in sorted(links, key=lambda x: -x['confidence']):
    key = (l['source'], l['target'])
    if key not in seen:
        seen.add(key)
        deduped.append(l)

# Limit per target
target_counts = defaultdict(int)
final_links = []
for l in deduped:
    if target_counts[l['target']] < 5:
        final_links.append(l)
        target_counts[l['target']] += 1

print(f"Total links: {len(links)}")
print(f"After dedup: {len(deduped)}")
print(f"After limiting (5 per buyer): {len(final_links)}")

type_counts = Counter(l['type'] for l in final_links)
for t, n in type_counts.most_common():
    print(f"  {t}: {n}")

# ════════════════════════════════════════════════════════════
# BUILD OUTPUT
# ════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("BUILDING OUTPUT")
print("=" * 60)

# District summary
dist_summary = {}
for dist in set(r['dist'] for r in rows):
    drows = [r for r in rows if r['dist'] == dist]
    lats = [r['lat'] for r in drows if 15 < r['lat'] < 20]
    lons = [r['lon'] for r in drows if 77 < r['lon'] < 82]
    if not lats:
        continue

    cats_count = Counter(r['cat'] for r in drows)
    pcbs_count = Counter(r['pcb'] for r in drows)
    dist_links = [l for l in final_links if l['district'] == dist]

    # Mandal clusters in this district
    mandal_clusters = []
    for (d, m), companies in mandal_groups.items():
        if d == dist and len(companies) >= 10:
            mandal_clusters.append({
                'mandal': m,
                'count': len(companies),
                'anchors': sum(1 for c in companies if c['cat'] >= 3),
                'estate_pct': round(sum(c['estate'] for c in companies) / len(companies), 2),
                'top': sorted(companies, key=lambda x: -x['inv'])[0]['name'][:35],
            })
    mandal_clusters.sort(key=lambda x: -x['count'])

    dist_summary[dist] = {
        'center': [round(sum(lats)/len(lats), 4), round(sum(lons)/len(lons), 4)],
        'count': len(drows),
        'total_inv': round(sum(r['inv'] for r in drows), 1),
        'total_emp': sum(r['emp'] for r in drows),
        'categories': {str(k): v for k, v in cats_count.items()},
        'pcb_tiers': {str(k): v for k, v in pcbs_count.items()},
        'estate_count': sum(r['estate'] for r in drows),
        'link_count': len(dist_links),
        'mandal_clusters': mandal_clusters[:8],
        'top_companies': [
            {'id': c['id'], 'name': c['name'][:40], 'inv': c['inv'], 'emp': c['emp'], 'cat': c['cat']}
            for c in sorted(drows, key=lambda x: -x['inv'])[:5]
        ],
    }

# Cluster profiles
econ_profiles = []
for ci in range(best_k):
    cl_rows = [rows[i] for i in range(len(rows)) if labels[i] == ci]
    avg_inv = np.mean([r['inv'] for r in cl_rows])
    avg_emp = np.mean([r['emp'] for r in cl_rows])
    cats = Counter(r['cat'] for r in cl_rows)
    pcbs = Counter(r['pcb'] for r in cl_rows)
    estate_pct = sum(r['estate'] for r in cl_rows) / len(cl_rows)

    if avg_inv > 1000 and avg_emp > 100:
        profile = "Large industrial"
    elif avg_inv > 100 and estate_pct > 0.3:
        profile = "Estate-based mid"
    elif avg_emp > 20 and avg_inv < 100:
        profile = "Labor-intensive"
    elif avg_inv < 30 and avg_emp < 10:
        profile = "Micro home-based"
    else:
        profile = "Mixed small"

    econ_profiles.append({
        'id': ci,
        'name': profile,
        'count': len(cl_rows),
        'avg_inv': round(avg_inv, 1),
        'avg_emp': round(avg_emp, 1),
        'estate_pct': round(estate_pct, 2),
        'dominant_cat': CATS[cats.most_common(1)[0][0]],
    })

output = {
    'districts': dist_summary,
    'links': final_links,
    'link_types': {
        'size_hierarchy': 'Larger company in same mandal likely buys from smaller companies nearby',
        'estate_colocation': 'Both companies in same industrial estate — designed for supply chain clustering',
        'odop_chain': 'Both ODOP companies contributing to district specialty product chain',
    },
    'econ_profiles': econ_profiles,
    'mandal_clusters': len(clusters),
    'stats': {
        'total_companies': len(rows),
        'total_links': len(final_links),
        'districts': len(dist_summary),
        'mandal_clusters': len(clusters),
        'link_type_counts': dict(type_counts),
    },
}

out_path = os.path.join(os.path.dirname(__file__), '..', 'msme-dashboard', 'src', 'relationships.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f)

size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"\nExported to {out_path}")
print(f"File size: {size_mb:.1f} MB")
print(f"Districts: {len(dist_summary)}")
print(f"Links: {len(final_links)}")
print(f"Mandal clusters: {len(clusters)}")
print(f"Economic profiles: {len(econ_profiles)}")
