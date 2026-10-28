import csv, json, os

CATS = {'MICRO':0,'Micro':0,'SMALL':1,'Small':1,'MEDIUM':2,'Medium':2,'LARGE':3,'Large':3,'MEGA':4,'Mega':4}
PCBS = {'GREEN':0,'Green':0,'ORANGE':1,'Orange':1,'RED':2,'Red':2,'WHITE':3,'White':3}
SOCS = {'OC':0,'OBC':1,'SC':2,'ST':3,'':4}

districts = []
mandals = []
d_map = {}
m_map = {}
units = []

with open('MSME_new.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dist = (row.get('DISTRICT_NAME') or '').strip() or 'Unknown'
        mand = (row.get('mandal_name') or '').strip() or 'Unknown'

        if dist not in d_map:
            d_map[dist] = len(districts)
            districts.append(dist)
        if mand not in m_map:
            m_map[mand] = len(mandals)
            mandals.append(mand)

        cat_raw = (row.get('CATEGOERY_NAME') or '').strip()
        pcb_raw = (row.get('PCBCATEGORY') or '').strip()
        soc_raw = (row.get('social_status') or '').strip()

        try: inv = float(row['investment'])
        except: inv = 0
        try: emp = int(float(row['employment']))
        except: emp = 0
        try: turn = float(row['turnover'])
        except: turn = 0
        try: lat = float(row['latitude'])
        except: lat = 0
        try: lon = float(row['longitude'])
        except: lon = 0

        park = (row.get('industrial_park') or '')
        estate = 1 if 'In Industrial' in park and 'Not' not in park else 0
        women = 1 if (row.get('promoter_women') or '').strip().upper() == 'YES' else 0
        odop = 1 if (row.get('ODOP') or '').strip().upper() == 'YES' else 0

        try: sl = int(row['SL'])
        except: sl = len(units) + 1

        name = (row.get('UNIT_NAME') or '').strip()
        name = name.replace('\\', '').replace('"', "'")

        units.append([
            sl, name, d_map[dist], m_map[mand],
            CATS.get(cat_raw, 0), PCBS.get(pcb_raw, 3), SOCS.get(soc_raw, 4),
            round(inv, 1), emp, round(turn, 1),
            estate, women, odop,
            round(lat, 5), round(lon, 5)
        ])

units.sort(key=lambda u: -u[7])

final = units
final.sort(key=lambda u: -u[7])

print(f'Total units: {len(units)}')
print(f'Final: {len(final)}')
print(f'Districts: {len(districts)}')
print(f'Mandals: {len(mandals)}')

with open('msme-dashboard/src/data.js', 'w', encoding='utf-8') as out:
    out.write('// Auto-generated MSME dataset\n')
    out.write(f'export const districts = {json.dumps(districts)};\n')
    out.write(f'export const mandals = {json.dumps(mandals)};\n')
    out.write(f'export const units = {json.dumps(final)};\n')
    out.write(f'export const totalMSMEs = {len(units)};\n')

size = os.path.getsize('msme-dashboard/src/data.js') / 1024 / 1024
print(f'data.js size: {size:.1f} MB')
