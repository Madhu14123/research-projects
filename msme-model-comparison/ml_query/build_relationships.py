"""
Build company-to-company relationship graph from MSME data.
Infers supply chain links from sector compatibility, geography, and size hierarchy.
Exports a JSON file the React app loads.
"""
import csv, json, os, sys
from collections import defaultdict
import math

sys.stdout.reconfigure(encoding='utf-8')

# ── SECTOR DETECTION (same as frontend) ──
SECTOR_KEYS = {
    "health": ["pharma","hospital","medical","health","clinic","drug","biotech","surgical","diagnos","laborator","ayur","herbal","capsule","tablet","syringe","vaccine","therap","oncol","formul","lifesci"],
    "textile": ["textile","garment","fabric","cotton","spinning","weaving","apparel","cloth","yarn","denim","knit","stitch","loom","dyeing","silk","jute","hosiery"],
    "food": ["food","rice","flour","dairy","milk","poultry","bakery","spice","oil mill","chicken","meat","sugar","grain","cattle","feed","fish","aqua","prawn","edible","masala","biscuit","noodle","snack","beverage","water plant","ice cream","paraboil","parboil"],
    "construction": ["cement","brick","concrete","steel","construction","granite","marble","sand ","stone","tiles","readymix","rmc ","fly ash","aac block","lime","gypsum","roofing","plumb","infra"],
    "auto": ["auto","engine","motor","vehicle","battery","machiner","casting","forg","weld","fabricat","cnc","lathe","precision","gear","bearing","piston","axle","chassis","tractor","two wheeler"],
    "energy": ["solar","power plant","energy","wind ","biomass","generat","turbine","transform","inverter","lithium","cell tech"],
    "chemical": ["chemical","paint","plastic","polymer","rubber","fertiliz","pesticide","resin","pigment","adhesive","solvent","acid","coating","ink ","pvc","hdpe","fibre glass"],
    "it": ["software","tech park","data center","semicon","chip","electron","circuit","iot","sensor","display","computer","server","infosys","wipro"],
    "packaging": ["packag","paper","carton","corrugat","box ","printing","label","pouch","wrapper","cardboard","pulp"],
    "furniture": ["furniture","wood","plywood","timber","laminate","veneer","mdf","particl","carpent"],
}

# ── SECTOR-TO-SECTOR SUPPLY CHAIN ──
# Each entry: (supplier_sector, buyer_sector, strength)
# strength: 1.0 = direct supply chain, 0.5 = moderate, 0.3 = weak
SUPPLY_CHAINS = [
    ("chemical", "health", 1.0),      # chemicals → pharma (APIs, excipients)
    ("chemical", "food", 0.7),        # chemicals → food (preservatives, additives)
    ("chemical", "textile", 0.8),     # chemicals → textile (dyes, finishing)
    ("chemical", "construction", 0.6),# chemicals → construction (paints, coatings)
    ("chemical", "auto", 0.5),        # chemicals → auto (lubricants, coatings)
    ("packaging", "health", 0.9),     # packaging → pharma (blister packs, boxes)
    ("packaging", "food", 0.9),       # packaging → food (containers, wraps)
    ("packaging", "chemical", 0.6),   # packaging → chemical (drums, containers)
    ("packaging", "textile", 0.5),    # packaging → textile (shipping)
    ("auto", "construction", 0.7),    # auto/engineering → construction (machinery)
    ("auto", "food", 0.4),            # auto → food (processing machines)
    ("construction", "auto", 0.6),    # steel/construction → auto (raw materials)
    ("construction", "energy", 0.5),  # construction → energy (infrastructure)
    ("energy", "health", 0.4),        # energy → pharma (power supply)
    ("energy", "construction", 0.5),  # energy → construction (power)
    ("energy", "auto", 0.5),          # energy → auto (EV batteries)
    ("textile", "food", 0.3),         # textile → food (jute bags)
    ("furniture", "construction", 0.6),# furniture → construction (interiors)
    ("it", "health", 0.4),            # IT → pharma (data systems)
    ("it", "auto", 0.4),              # IT → auto (embedded systems)
    ("food", "packaging", 0.3),       # food waste → packaging (bio packaging)
]

def detect_sector(name):
    name_lower = name.lower()
    for sec, keys in SECTOR_KEYS.items():
        if any(k in name_lower for k in keys):
            return sec
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# ── LOAD DATA ──
BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, '..', 'MSME_new.csv')

CATS = {"MICRO": 0, "Micro": 0, "SMALL": 1, "Small": 1, "MEDIUM": 2, "Medium": 2, "LARGE": 3, "Large": 3, "MEGA": 4, "Mega": 4}
PCBS = {"GREEN": 0, "Green": 0, "ORANGE": 1, "Orange": 1, "RED": 2, "Red": 2, "WHITE": 3, "White": 3}

companies = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        try:
            sl = int(row['SL'])
            name = (row.get('UNIT_NAME') or '').strip()
            dist = (row.get('DISTRICT_NAME') or '').strip()
            mandal = (row.get('mandal_name') or '').strip()
            cat = CATS.get((row.get('CATEGOERY_NAME') or '').strip(), 0)
            pcb = PCBS.get((row.get('PCBCATEGORY') or '').strip(), 3)
            inv = float(row['investment']) if row.get('investment') else 0
            emp = int(float(row['employment'])) if row.get('employment') else 0
            lat = float(row['latitude']) if row.get('latitude') else 0
            lon = float(row['longitude']) if row.get('longitude') else 0
            sector = detect_sector(name)

            if inv > 0:  # skip zero-investment entries
                companies.append({
                    'id': sl, 'name': name, 'district': dist, 'mandal': mandal,
                    'cat': cat, 'pcb': pcb, 'inv': inv, 'emp': emp,
                    'lat': lat, 'lon': lon, 'sector': sector,
                })
        except:
            pass

print(f"Loaded {len(companies)} companies with investment > 0")

# Sector distribution
sector_counts = defaultdict(int)
for c in companies:
    sector_counts[c['sector'] or 'unknown'] += 1
for s, n in sorted(sector_counts.items(), key=lambda x: -x[1]):
    print(f"  {s}: {n}")

# ── BUILD SUPPLY CHAIN LOOKUP ──
chain_lookup = {}
for sup, buy, strength in SUPPLY_CHAINS:
    chain_lookup[(sup, buy)] = strength

# ── INFER RELATIONSHIPS ──
# Strategy: For each district, find plausible buyer-supplier pairs
# Only link companies that have:
# 1. Compatible sectors (from supply chain graph)
# 2. Size hierarchy (buyer >= supplier tier + 1)
# 3. Same district or mandal

print("\nInferring relationships...")

# Index companies by district
by_district = defaultdict(list)
for c in companies:
    if c['district']:
        by_district[c['district']].append(c)

links = []
link_id = 0

for dist, dist_companies in by_district.items():
    # Only process companies with known sectors
    sectored = [c for c in dist_companies if c['sector']]

    # For each pair of sectored companies in same district
    for i, buyer in enumerate(sectored):
        if buyer['cat'] < 2:  # buyers should be at least MEDIUM
            continue
        for supplier in sectored:
            if supplier['id'] == buyer['id']:
                continue
            if supplier['cat'] >= buyer['cat']:  # supplier should be smaller
                continue

            # Check sector compatibility
            chain_key = (supplier['sector'], buyer['sector'])
            strength = chain_lookup.get(chain_key, 0)
            if strength == 0:
                continue

            # Geographic bonus
            geo_score = 0.7  # same district
            if supplier['mandal'] == buyer['mandal'] and supplier['mandal']:
                geo_score = 1.0  # same mandal

            # Size compatibility: best when buyer is 1-2 tiers above
            tier_diff = buyer['cat'] - supplier['cat']
            size_score = 1.0 if tier_diff <= 2 else 0.5

            # Investment ratio: supplier should be smaller but not tiny
            if buyer['inv'] > 0 and supplier['inv'] > 0:
                ratio = supplier['inv'] / buyer['inv']
                inv_score = 1.0 if 0.01 < ratio < 0.5 else 0.5
            else:
                inv_score = 0.3

            # Combined confidence
            confidence = round(strength * geo_score * size_score * inv_score, 3)

            if confidence >= 0.15:  # minimum threshold
                links.append({
                    'id': link_id,
                    'source': supplier['id'],
                    'target': buyer['id'],
                    'type': 'supply',
                    'supplier_sector': supplier['sector'],
                    'buyer_sector': buyer['sector'],
                    'confidence': confidence,
                    'same_mandal': supplier['mandal'] == buyer['mandal'],
                    'district': dist,
                })
                link_id += 1

print(f"Total links inferred: {len(links)}")

# Too many links — keep only top N per buyer (most confident suppliers)
MAX_LINKS_PER_BUYER = 5
buyer_links = defaultdict(list)
for l in links:
    buyer_links[l['target']].append(l)

filtered_links = []
for buyer_id, blinks in buyer_links.items():
    blinks.sort(key=lambda x: -x['confidence'])
    filtered_links.extend(blinks[:MAX_LINKS_PER_BUYER])

print(f"After filtering (top {MAX_LINKS_PER_BUYER} per buyer): {len(filtered_links)}")

# Link type distribution
type_counts = defaultdict(int)
for l in filtered_links:
    type_counts[f"{l['supplier_sector']} -> {l['buyer_sector']}"] += 1
for t, n in sorted(type_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"  {t}: {n}")

# ── BUILD DISTRICT SUMMARY ──
dist_summary = {}
for dist, dcs in by_district.items():
    if not dist:
        continue
    lats = [c['lat'] for c in dcs if 15 < c['lat'] < 20]
    lons = [c['lon'] for c in dcs if 77 < c['lon'] < 82]
    if not lats:
        continue

    sector_breakdown = defaultdict(int)
    for c in dcs:
        sector_breakdown[c['sector'] or 'other'] += 1

    cat_breakdown = defaultdict(int)
    for c in dcs:
        cat_breakdown[c['cat']] += 1

    dist_links = [l for l in filtered_links if l['district'] == dist]

    dist_summary[dist] = {
        'center': [round(sum(lats)/len(lats), 4), round(sum(lons)/len(lons), 4)],
        'count': len(dcs),
        'total_inv': round(sum(c['inv'] for c in dcs), 1),
        'total_emp': sum(c['emp'] for c in dcs),
        'sectors': dict(sector_breakdown),
        'categories': {str(k): v for k, v in cat_breakdown.items()},
        'link_count': len(dist_links),
        'top_companies': [
            {'id': c['id'], 'name': c['name'][:40], 'inv': c['inv'], 'emp': c['emp'], 'cat': c['cat'], 'sector': c['sector']}
            for c in sorted(dcs, key=lambda x: -x['inv'])[:5]
        ],
    }

# ── EXPORT ──
output = {
    'districts': dist_summary,
    'links': filtered_links,
    'supply_chains': [{'from': s, 'to': b, 'strength': st} for s, b, st in SUPPLY_CHAINS],
    'stats': {
        'total_companies': len(companies),
        'total_links': len(filtered_links),
        'districts': len(dist_summary),
        'sectors': dict(sector_counts),
    },
}

out_path = os.path.join(BASE, '..', 'msme-dashboard', 'src', 'relationships.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f)

size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"\nExported to {out_path}")
print(f"File size: {size_mb:.1f} MB")
print(f"Districts: {len(dist_summary)}")
print(f"Links: {len(filtered_links)}")
