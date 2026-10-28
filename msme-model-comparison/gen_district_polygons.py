import json, re, math
from collections import defaultdict
from scipy.spatial import ConvexHull
import numpy as np

txt = open(r'D:\A_Design\msme-dashboard\src\data.js', encoding='utf-8').read()

dm = re.search(r'export const districts\s*=\s*(\[[^\]]+\])', txt)
districts = json.loads(dm.group(1))

start = txt.index('export const units = [') + len('export const units = ')
depth, i = 0, start
while i < len(txt):
    if txt[i] == '[': depth += 1
    elif txt[i] == ']':
        depth -= 1
        if depth == 0:
            end = i + 1; break
    i += 1
units = json.loads(txt[start:end])
print(f'Units: {len(units)}')

# Collect GPS coords per district
dist_pts = defaultdict(list)
for u in units:
    did, lat, lon = u[2], u[13], u[14]
    if 15.5 < lat < 20.5 and 76.5 < lon < 82.0 and did < len(districts):
        dist_pts[did].append([lat, lon])

print(f'Districts with pts: {len(dist_pts)}')

# Compute convex hull per district, expand slightly for overlap
dist_polys = {}
for did, pts in dist_pts.items():
    arr = np.array(pts)
    if len(arr) < 4:
        continue
    hull = ConvexHull(arr)
    verts = arr[hull.vertices]
    # expand hull outward by ~0.05 deg from centroid
    cen = verts.mean(axis=0)
    expanded = cen + (verts - cen) * 1.08
    # round to 4 dp
    poly = [[round(float(p[0]),4), round(float(p[1]),4)] for p in expanded]
    dist_polys[did] = {'name': districts[did], 'poly': poly}
    print(f'  [{did:2d}] {districts[did]:25s}: {len(poly)} vertices')

# Write as JS
out = '// Auto-generated district convex hull polygons (from company GPS data)\n'
out += 'export const DIST_POLYS = {\n'
for did in sorted(dist_polys.keys()):
    d = dist_polys[did]
    pts_str = ','.join(f'[{p[0]},{p[1]}]' for p in d['poly'])
    out += f'  {did}: [{pts_str}],  // {d["name"]}\n'
out += '};\n'

open(r'D:\A_Design\msme-dashboard\src\dist_polys.js', 'w', encoding='utf-8').write(out)
sz = len(out)
print(f'\nWrote dist_polys.js ({sz} bytes, {len(dist_polys)} districts)')
