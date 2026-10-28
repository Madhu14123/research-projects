import json, math, re

data = json.load(open(r'D:\A_Design\telangana_districts.geojson', encoding='utf-8'))

# Known Telangana district name substrings
TG_NAMES = {
    'adilabad','komaram','mancherial','nirmal','nizamabad','jagtial','rajanna','sircilla',
    'karimnagar','peddapalli','jayashankar','bhupalpally','mulugu','bhadradri','khammam',
    'nalgonda','suryapet','yadadri','bhongiri','medchal','hydrabad','hyderabad',
    'rangareddy','vikarabad','sangareddy','medak','siddipet','kamareddy',
    'wanaparthy','mahabubnagar','mahbubnagar','narayanpet','jogulamba','nagarkurnool',
    'warangal','jangaon','mahabubabad','hanumakonda'
}

def is_tg(name):
    n = name.lower()
    return any(k in n for k in TG_NAMES)

# Simplify polygon: keep every Nth point (Douglas-Peucker would be better but this is fast)
def simplify(coords, target=60):
    if len(coords) <= target:
        return coords
    step = max(1, len(coords) // target)
    simplified = coords[::step]
    if simplified[-1] != coords[-1]:
        simplified.append(coords[-1])
    return simplified

def extract_rings(geom):
    """Return list of coordinate rings from Polygon/MultiPolygon"""
    rings = []
    if geom['type'] == 'Polygon':
        rings.append(geom['coordinates'][0])  # outer ring only
    elif geom['type'] == 'MultiPolygon':
        # take largest polygon
        best = max(geom['coordinates'], key=lambda p: len(p[0]))
        rings.append(best[0])
    return rings

tg_features = [f for f in data['features'] if is_tg(f['properties']['shapeName'])]
print(f'Filtered to {len(tg_features)} Telangana districts:')

dist_js = {}
for f in tg_features:
    name = f['properties']['shapeName']
    print(f'  {name}')
    rings = extract_rings(f['geometry'])
    for ring in rings:
        simp = simplify(ring, 80)
        # coords are [lon, lat] in GeoJSON
        pts = [[round(c[1],5), round(c[0],5)] for c in simp]  # convert to [lat, lon]
        dist_js[name] = pts

print(f'\nTotal districts: {len(dist_js)}')

# Compute bounding box of all TG districts
all_lats = [p[0] for pts in dist_js.values() for p in pts]
all_lons = [p[1] for pts in dist_js.values() for p in pts]
print(f'Telangana bounds: lat [{min(all_lats):.3f}, {max(all_lats):.3f}] lon [{min(all_lons):.3f}, {max(all_lons):.3f}]')

# Write JS
out = '// Official Telangana district boundaries (geoBoundaries open data)\n'
out += 'export const TG_DISTRICTS = {\n'
for name, pts in sorted(dist_js.items()):
    pts_str = ','.join(f'[{p[0]},{p[1]}]' for p in pts)
    out += f'  {json.dumps(name)}: [{pts_str}],\n'
out += '};\n'

open(r'D:\A_Design\msme-dashboard\src\tg_districts.js', 'w', encoding='utf-8').write(out)
print(f'Wrote tg_districts.js ({len(out)} bytes)')
