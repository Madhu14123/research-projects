import json, re, math
from collections import defaultdict

txt = open(r'D:\A_Design\msme-dashboard\src\data.js', encoding='utf-8').read()
rel = json.load(open(r'D:\A_Design\msme-dashboard\src\relationships.json', encoding='utf-8'))

dm = re.search(r'export const districts\s*=\s*(\[[^\]]+\])', txt)
districts = json.loads(dm.group(1))

# Parse units — format: [id, "name", distIdx, mandalIdx, catIdx, pcbIdx, socIdx, inv, emp, turn, estate, women, odop, lat, lon]
# Use json to parse the units array
units_match = re.search(r'export const units\s*=\s*(\[.+?\]);', txt, re.S)
if not units_match:
    # try without semicolon
    start = txt.index('export const units = [') + len('export const units = ')
    # find balanced closing bracket
    depth, i = 0, start
    while i < len(txt):
        if txt[i] == '[': depth += 1
        elif txt[i] == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    units_json = txt[start:end]
else:
    units_json = units_match.group(1)

print('Parsing units...')
units = json.loads(units_json)
print(f'Total units: {len(units)}')
print('Sample:', units[0])

# Build lookup
dist_coords = defaultdict(list)
unit_map = {}
for u in units:
    uid, did, cat, lat, lon = u[0], u[2], u[4], u[13], u[14]
    if 15.5 < lat < 20.5 and 76.5 < lon < 82.0:
        dist_coords[did].append((lat, lon))
        unit_map[uid] = (did, cat, lat, lon)

print(f'Units with valid GPS: {len(unit_map)}')

# District centroids
centroids = {}
for did, coords in dist_coords.items():
    if did < len(districts) and len(coords) >= 3:
        la = sum(c[0] for c in coords) / len(coords)
        lo = sum(c[1] for c in coords) / len(coords)
        centroids[did] = {'lat': la, 'lon': lo, 'n': len(coords), 'name': districts[did]}

print(f'Districts with centroids: {len(centroids)}')
for did, c in sorted(centroids.items(), key=lambda x: -x[1]['n']):
    print(f'  [{did:2d}] {c["name"]:25s}: ({c["lat"]:.4f}, {c["lon"]:.4f}) n={c["n"]}')

# Select meaningful verifiable links
links = rel['links']

# estate_colocation = most verifiable (physically co-located in industrial estate)
estate_links = [l for l in links if l['type'] == 'estate_colocation']
size_links   = [l for l in links if l['type'] == 'size_hierarchy' and l.get('confidence', 0) >= 0.9]
odop_links   = [l for l in links if l['type'] == 'odop_chain']

def enrich(lst):
    out = []
    for l in lst:
        s, t = l['source'], l['target']
        if s in unit_map and t in unit_map:
            slat, slon = unit_map[s][2], unit_map[s][3]
            tlat, tlon = unit_map[t][2], unit_map[t][3]
            out.append({
                'src': s, 'tgt': t,
                'slat': slat, 'slon': slon,
                'tlat': tlat, 'tlon': tlon,
                'conf': l.get('confidence', 0.8),
                'type': l['type'],
                'district': l.get('district', ''),
            })
    return out

estate_valid = enrich(sorted(estate_links, key=lambda x: -x.get('confidence', 0)))
size_valid   = enrich(sorted(size_links,   key=lambda x: -x.get('confidence', 0)))
odop_valid   = enrich(sorted(odop_links,   key=lambda x: -x.get('confidence', 0)))

print(f'\nValid estate links: {len(estate_valid)}')
print(f'Valid size links:   {len(size_valid)}')
print(f'Valid odop links:   {len(odop_valid)}')

# Pick top 3 per district per type, max 60 total
def pick_spread(pool, per_dist=3):
    by_d = defaultdict(list)
    for l in pool:
        by_d[l['district']].append(l)
    out = []
    for d, lks in sorted(by_d.items(), key=lambda x: -len(x[1])):
        out.extend(lks[:per_dist])
    return out

selected_estate = pick_spread(estate_valid, 3)
selected_size   = pick_spread(size_valid,   2)
selected_odop   = pick_spread(odop_valid,   2)

selected = selected_estate + selected_size + selected_odop
print(f'Total selected links: {len(selected)}')

# Output JS
out = '// Auto-generated map data\n'
out += 'export const MAP_LINKS = [\n'
for l in selected:
    out += f'  {{src:{l["src"]},tgt:{l["tgt"]},slat:{l["slat"]},slon:{l["slon"]},tlat:{l["tlat"]},tlon:{l["tlon"]},conf:{l["conf"]},type:"{l["type"]}",district:"{l["district"]}"}},\n'
out += '];\n\n'

out += 'export const DIST_CENTROIDS = {\n'
for did in sorted(centroids.keys()):
    c = centroids[did]
    out += f'  {did}: [{c["lat"]:.4f},{c["lon"]:.4f},{c["n"]}],  // {c["name"]}\n'
out += '};\n'

open(r'D:\A_Design\msme-dashboard\src\map_data.js', 'w', encoding='utf-8').write(out)
print('\nWrote map_data.js')
