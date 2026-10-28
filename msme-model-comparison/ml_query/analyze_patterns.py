"""Deep pattern analysis of the MSME dataset — find real structures before visualizing."""
import csv, sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

SECTOR_KEYS = {
    'health': ['pharma','hospital','medical','health','clinic','drug','biotech','surgical','diagnos','laborator','ayur','herbal','capsule','tablet','vaccine','therap','formul'],
    'textile': ['textile','garment','fabric','cotton','spinning','weaving','apparel','cloth','yarn','denim','knit','loom','dyeing','silk','jute','hosiery'],
    'food': ['food','rice','flour','dairy','milk','poultry','bakery','spice','oil mill','chicken','meat','sugar','grain','feed','fish','aqua','prawn','edible','masala','biscuit','paraboil','parboil'],
    'construction': ['cement','brick','concrete','steel','construction','granite','marble','tiles','readymix','fly ash','aac block','lime','gypsum','roofing'],
    'auto': ['auto','engine','motor','vehicle','battery','machiner','casting','forg','weld','fabricat','cnc','lathe','precision','gear','bearing','piston','axle','tractor'],
    'energy': ['solar','power plant','energy','wind ','biomass','turbine','transform','inverter','lithium','cell tech'],
    'chemical': ['chemical','paint','plastic','polymer','rubber','fertiliz','pesticide','resin','pigment','adhesive','coating','pvc','hdpe'],
    'it': ['software','tech park','data center','semicon','chip','electron','circuit','iot','sensor','computer','server'],
    'packaging': ['packag','paper','carton','corrugat','printing','label','pouch','cardboard','pulp'],
    'furniture': ['furniture','wood','plywood','timber','laminate','veneer','mdf','carpent'],
}

CATS_MAP = {'MICRO':0,'Micro':0,'SMALL':1,'Small':1,'MEDIUM':2,'Medium':2,'LARGE':3,'Large':3,'MEGA':4,'Mega':4}
CATS = ['MICRO','SMALL','MEDIUM','LARGE','MEGA']

def detect(name):
    nl = name.lower()
    for s, keys in SECTOR_KEYS.items():
        if any(k in nl for k in keys):
            return s
    return 'other'

rows = []
with open('MSME_new.csv', 'r', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        name = (r.get('UNIT_NAME') or '').strip()
        dist = (r.get('DISTRICT_NAME') or '').strip()
        mandal = (r.get('mandal_name') or '').strip()
        cat = CATS_MAP.get((r.get('CATEGOERY_NAME') or '').strip(), 0)
        try: inv = float(r['investment'])
        except: inv = 0
        try: emp = int(float(r['employment']))
        except: emp = 0
        sec = detect(name)
        if dist and inv > 0:
            rows.append({'name':name,'dist':dist,'mandal':mandal,'cat':cat,'inv':inv,'emp':emp,'sec':sec})

print(f"=== DATASET: {len(rows)} companies ===\n")

# PATTERN 1: Sector distribution
print("PATTERN 1: SECTOR DISTRIBUTION")
sec_stats = defaultdict(lambda: {'n':0,'inv':0,'emp':0})
for r in rows:
    sec_stats[r['sec']]['n'] += 1
    sec_stats[r['sec']]['inv'] += r['inv']
    sec_stats[r['sec']]['emp'] += r['emp']
for s, v in sorted(sec_stats.items(), key=lambda x:-x[1]['n']):
    print(f"  {s:15s}: {v['n']:6d} companies, inv={v['inv']/100000:.0f}Cr, emp={v['emp']:,}")
print()

# PATTERN 2: Size tiers per sector
print("PATTERN 2: SIZE TIERS PER SECTOR")
for s in ['health','food','construction','auto','chemical','textile','energy','it','packaging','furniture']:
    tier = Counter(r['cat'] for r in rows if r['sec']==s)
    total = sum(tier.values())
    if total == 0: continue
    pcts = [f"{tier.get(i,0)/total*100:.0f}%" for i in range(5)]
    print(f"  {s:15s}: MI={pcts[0]:>4s} SM={pcts[1]:>4s} MD={pcts[2]:>4s} LG={pcts[3]:>4s} MG={pcts[4]:>4s}  (n={total})")
print()

# PATTERN 3: District specialization
print("PATTERN 3: DISTRICT SPECIALIZATION (top 3 per sector)")
for s in ['health','food','construction','auto','chemical','textile','energy','packaging']:
    dist_count = Counter(r['dist'] for r in rows if r['sec']==s)
    top3 = dist_count.most_common(3)
    total = sum(dist_count.values())
    print(f"  {s:15s}: {' | '.join(f'{d}({n})' for d,n in top3)}  [{total} total]")
print()

# PATTERN 4: Mandal clusters
print("PATTERN 4: MANDAL-SECTOR CLUSTERS (5+ same-sector companies)")
mandal_sec = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r['sec'] != 'other':
        mandal_sec[(r['dist'],r['mandal'])][r['sec']].append(r)
clusters = []
for (dist,mandal), secs in mandal_sec.items():
    for sec, companies in secs.items():
        if len(companies) >= 5:
            total_inv = sum(c['inv'] for c in companies)
            total_emp = sum(c['emp'] for c in companies)
            cats = Counter(c['cat'] for c in companies)
            clusters.append((dist, mandal, sec, len(companies), total_inv, total_emp, cats))
clusters.sort(key=lambda x: -x[3])
print(f"  Found {len(clusters)} clusters")
for dist, mandal, sec, n, inv, emp, cats in clusters[:20]:
    tier_str = '+'.join(f"{CATS[k][0]}{v}" for k,v in sorted(cats.items()))
    print(f"    {dist:15s}/{mandal:15s}: {sec:12s} {n:3d} co. inv={inv/100:.0f}L emp={emp:,} [{tier_str}]")
print()

# PATTERN 5: Buyer-supplier ratio
print("PATTERN 5: BUYER-SUPPLIER RATIO (LARGE+MEGA vs rest)")
dist_tiers = defaultdict(lambda: {'b':0,'s':0,'bi':0,'si':0})
for r in rows:
    if r['cat'] >= 3:
        dist_tiers[r['dist']]['b'] += 1
        dist_tiers[r['dist']]['bi'] += r['inv']
    else:
        dist_tiers[r['dist']]['s'] += 1
        dist_tiers[r['dist']]['si'] += r['inv']
for d, v in sorted(dist_tiers.items(), key=lambda x: -x[1]['b'])[:15]:
    ratio = v['b']/max(v['s'],1)
    gap = "DEFICIT" if ratio > 0.15 else "SURPLUS" if ratio < 0.03 else "BALANCED"
    print(f"  {d:18s}: {v['b']:4d} buyers + {v['s']:5d} suppliers  ratio={ratio:.3f}  {gap}")
print()

# PATTERN 6: Cross-sector co-location
print("PATTERN 6: SECTOR CO-LOCATION (appear together in same mandal, 3+ each)")
coloc = Counter()
for (dist,mandal), secs in mandal_sec.items():
    sec_list = [s for s in secs if len(secs[s]) >= 3]
    for i in range(len(sec_list)):
        for j in range(i+1, len(sec_list)):
            pair = tuple(sorted([sec_list[i], sec_list[j]]))
            coloc[pair] += 1
for (s1,s2), n in coloc.most_common(12):
    print(f"  {s1:12s} + {s2:12s}: co-located in {n} mandals")
print()

# PATTERN 7: Investment concentration
print("PATTERN 7: INVESTMENT CONCENTRATION (top 10 companies per sector)")
for s in ['health','auto','construction','chemical','energy']:
    sec_rows = sorted([r for r in rows if r['sec']==s], key=lambda x: -x['inv'])
    total_inv = sum(r['inv'] for r in sec_rows)
    top10_inv = sum(r['inv'] for r in sec_rows[:10])
    print(f"  {s:15s}: top 10 hold {top10_inv/max(total_inv,1)*100:.0f}% of sector investment")
    for r in sec_rows[:3]:
        print(f"    {r['name'][:40]:42s} {r['dist']:15s} inv={r['inv']/100:.0f}L cat={CATS[r['cat']]}")
print()

# PATTERN 8: What percentage of companies have NO detectable sector?
unsectored = [r for r in rows if r['sec'] == 'other']
print(f"PATTERN 8: UNSECTORED COMPANIES: {len(unsectored)} / {len(rows)} ({len(unsectored)/len(rows)*100:.0f}%)")
# Sample names from unsectored
import random
random.seed(42)
sample = random.sample(unsectored, min(20, len(unsectored)))
print("  Sample unsectored names:")
for r in sample:
    print(f"    {r['name'][:50]:52s} {r['dist']:15s} {CATS[r['cat']]:6s} inv={r['inv']}")
