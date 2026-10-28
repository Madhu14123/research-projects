"""Generate synthetic NL query → structured output training data for MSME queries."""
import json, random, os, csv

random.seed(42)

# Load real district/mandal names from dataset
districts = []
mandals_by_dist = {}
with open(os.path.join(os.path.dirname(__file__), '..', 'MSME_new.csv'), 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        d = (row.get('DISTRICT_NAME') or '').strip()
        m = (row.get('mandal_name') or '').strip()
        if d and d not in districts:
            districts.append(d)
        if d and m:
            mandals_by_dist.setdefault(d, set()).add(m)

districts = [d for d in districts if d][:33]

INTENTS = ["search", "ranking", "comparison", "supplier_search", "buyer_search",
           "risk_assessment", "cluster_analysis", "diversity", "odop_analysis", "acquisition", "supply_demand"]

SECTORS = {
    "health": ["pharma", "medical", "health", "hospital", "biotech", "healthcare", "drug", "surgical", "diagnostic", "medicine"],
    "textile": ["textile", "garment", "fabric", "cotton", "spinning", "apparel", "cloth", "weaving"],
    "food": ["food", "rice", "dairy", "poultry", "bakery", "agri", "agriculture", "flour", "spice", "grain"],
    "construction": ["cement", "brick", "concrete", "steel", "construction", "granite", "tiles", "building"],
    "auto": ["auto", "automobile", "engineering", "motor", "vehicle", "battery", "casting", "machiner", "fabricat"],
    "energy": ["solar", "power", "energy", "electric", "wind", "renewable", "biomass"],
    "chemical": ["chemical", "paint", "plastic", "polymer", "rubber", "fertiliz", "pesticide"],
    "it": ["software", "electronics", "semiconductor", "computer", "it company", "tech company"],
    "packaging": ["packaging", "paper", "carton", "printing", "corrugat"],
    "furniture": ["furniture", "wood", "plywood", "timber", "laminate"],
}

CATEGORIES = ["MICRO", "SMALL", "MEDIUM", "LARGE", "MEGA"]
PCB_TIERS = ["GREEN", "ORANGE", "RED", "WHITE"]
SORT_OPTIONS = ["inv", "emp", "turn", "name", "inv_asc"]

COMPANY_NAMES = [
    "Amara Raja", "Dr. Reddy", "Aurobindo", "Kitex", "Ramagundam Fertilizers",
    "Hetero", "Natco", "Granules", "Laurus", "Sagar Cement", "Orient Cement",
    "Mahindra", "MRF", "HPCL", "ITC Limited", "Welspun", "Bharath Electronics", "Sanofi",
    "Unique Biotech", "MSN Laboratories",
]

data = []

def make_label(intent, sector=None, district=None, category=None, categories=None,
               pcb=None, women=False, odop=False, outside_estate=False, in_estate=False,
               min_investment=None, min_turnover=None, min_employment=None,
               limit=None, sort_by="inv", target_company=None, exclude_district=None, exclude_category=None):
    label = {"intent": intent}
    if sector: label["sector"] = sector
    if district: label["district"] = district
    if category: label["category"] = category
    if categories: label["categories"] = categories
    if pcb: label["pcb"] = pcb
    if women: label["women"] = True
    if odop: label["odop"] = True
    if outside_estate: label["outside_estate"] = True
    if in_estate: label["in_estate"] = True
    if min_investment: label["min_investment"] = min_investment
    if min_turnover: label["min_turnover"] = min_turnover
    if min_employment: label["min_employment"] = min_employment
    if limit: label["limit"] = limit
    if sort_by != "inv": label["sort_by"] = sort_by
    if target_company: label["target_company"] = target_company
    if exclude_district: label["exclude_district"] = exclude_district
    if exclude_category: label["exclude_category"] = exclude_category
    return label

# ── TEMPLATE GENERATORS ──

def gen_search():
    templates = []
    # Sector + district
    for sec_key, sec_words in SECTORS.items():
        for w in sec_words:
            d = random.choice(districts)
            templates.append((f"find all {w} companies in {d}", make_label("search", sector=sec_key, district=d)))
            templates.append((f"show {w} companies", make_label("search", sector=sec_key)))
            templates.append((f"list {w} companies in {d}", make_label("search", sector=sec_key, district=d)))
            templates.append((f"which {w} companies are in {d}", make_label("search", sector=sec_key, district=d)))
            templates.append((f"get all {w} units", make_label("search", sector=sec_key)))
    # Category searches
    for cat in CATEGORIES:
        d = random.choice(districts)
        templates.append((f"find all {cat} companies", make_label("search", category=cat)))
        templates.append((f"show {cat} companies in {d}", make_label("search", category=cat, district=d)))
        templates.append((f"list {cat} units in {d}", make_label("search", category=cat, district=d)))
    # PCB
    for pcb in PCB_TIERS:
        templates.append((f"show {pcb} PCB companies", make_label("search", pcb=pcb)))
        d = random.choice(districts)
        templates.append((f"find {pcb} PCB tier companies in {d}", make_label("search", pcb=pcb, district=d)))
    # Investment thresholds
    for val, unit, lakhs in [(100, "crore", 10000), (500, "crore", 50000), (50, "lakhs", 50), (200, "lakhs", 200)]:
        templates.append((f"companies with investment above {val} {unit}", make_label("search", min_investment=lakhs)))
        d = random.choice(districts)
        templates.append((f"find companies in {d} with investment over {val} {unit}", make_label("search", district=d, min_investment=lakhs)))
    # Employment thresholds
    for emp in [100, 500, 1000]:
        templates.append((f"companies with more than {emp} employees", make_label("search", min_employment=emp)))
    # ODOP
    templates.append(("show all ODOP companies", make_label("odop_analysis", odop=True)))
    templates.append(("find ODOP units", make_label("odop_analysis", odop=True)))
    # Estate
    templates.append(("companies in industrial estate", make_label("search", in_estate=True)))
    templates.append(("companies outside industrial estate", make_label("search", outside_estate=True)))
    return templates

def gen_diversity():
    templates = []
    templates.append(("find women led companies", make_label("diversity", women=True)))
    templates.append(("show women-led companies", make_label("diversity", women=True)))
    templates.append(("give top women led companies", make_label("diversity", women=True)))
    templates.append(("women owned companies", make_label("diversity", women=True)))
    templates.append(("female led enterprises", make_label("diversity", women=True)))
    templates.append(("list all women promoted companies", make_label("diversity", women=True)))
    for d in random.sample(districts, min(10, len(districts))):
        templates.append((f"women led companies in {d}", make_label("diversity", women=True, district=d)))
    for cat in CATEGORIES:
        templates.append((f"women led {cat} companies", make_label("diversity", women=True, category=cat)))
    templates.append(("women led LARGE or MEDIUM companies", make_label("diversity", women=True, categories=["LARGE", "MEDIUM"])))
    templates.append(("find women-led LARGE or MEDIUM companies across all districts", make_label("diversity", women=True, categories=["LARGE", "MEDIUM"])))
    for sec_key, sec_words in SECTORS.items():
        w = random.choice(sec_words)
        templates.append((f"women led {w} companies", make_label("diversity", women=True, sector=sec_key)))
    # With limits
    for n in [3, 5, 10, 15]:
        templates.append((f"top {n} women led companies", make_label("diversity", women=True, limit=n)))
        templates.append((f"give only {n} women led companies", make_label("diversity", women=True, limit=n)))
        templates.append((f"show first {n} women owned units", make_label("diversity", women=True, limit=n)))
    return templates

def gen_ranking():
    templates = []
    for n in [5, 10, 15, 20]:
        templates.append((f"top {n} companies by investment", make_label("ranking", limit=n)))
        templates.append((f"top {n} companies by employment", make_label("ranking", limit=n, sort_by="emp")))
        templates.append((f"top {n} companies by turnover", make_label("ranking", limit=n, sort_by="turn")))
        templates.append((f"give me the top {n} biggest companies", make_label("ranking", limit=n)))
        templates.append((f"show the {n} largest companies", make_label("ranking", limit=n)))
    for cat in CATEGORIES:
        templates.append((f"which districts have most {cat} companies", make_label("ranking", category=cat)))
        templates.append((f"densest {cat} company concentration by district", make_label("ranking", category=cat)))
    for sec_key, sec_words in SECTORS.items():
        w = random.choice(sec_words)
        templates.append((f"top {w} companies by investment", make_label("ranking", sector=sec_key)))
        templates.append((f"biggest {w} companies", make_label("ranking", sector=sec_key)))
        templates.append((f"largest {w} firms by employment", make_label("ranking", sector=sec_key, sort_by="emp")))
        d = random.choice(districts)
        templates.append((f"top {w} companies in {d}", make_label("ranking", sector=sec_key, district=d)))
    templates.append(("which districts have the densest MEGA company concentration", make_label("ranking", category="MEGA")))
    templates.append(("rank districts by number of companies", make_label("ranking")))
    templates.append(("smallest companies by investment", make_label("ranking", sort_by="inv_asc")))
    return templates

def gen_comparison():
    templates = []
    for name in COMPANY_NAMES:
        templates.append((f"compare {name} and similar companies", make_label("comparison", target_company=name)))
        templates.append((f"compare {name} vs peers", make_label("comparison", target_company=name)))
        templates.append((f"benchmark {name} against similar firms", make_label("comparison", target_company=name)))
        templates.append((f"how does {name} compare", make_label("comparison", target_company=name)))
        templates.append((f"companies similar to {name}", make_label("comparison", target_company=name)))
    return templates

def gen_supplier_buyer():
    templates = []
    for name in COMPANY_NAMES:
        templates.append((f"find suppliers for {name}", make_label("supplier_search", target_company=name)))
        templates.append((f"who supplies to {name}", make_label("supplier_search", target_company=name)))
        templates.append((f"potential vendors for {name}", make_label("supplier_search", target_company=name)))
        templates.append((f"find buyers for {name}", make_label("buyer_search", target_company=name)))
        templates.append((f"who buys from {name}", make_label("buyer_search", target_company=name)))
    for sec_key, sec_words in SECTORS.items():
        w = random.choice(sec_words)
        templates.append((f"which companies supply to {w} sector", make_label("supplier_search", sector=sec_key)))
        templates.append((f"suppliers for {w} industry", make_label("supplier_search", sector=sec_key)))
    return templates

def gen_risk():
    templates = []
    templates.append(("show RED PCB risk companies outside industrial estates", make_label("risk_assessment", pcb="RED", outside_estate=True)))
    templates.append(("companies with environmental risk", make_label("risk_assessment")))
    templates.append(("high pollution companies outside estates", make_label("risk_assessment", outside_estate=True)))
    templates.append(("RED PCB companies not in estate", make_label("risk_assessment", pcb="RED", outside_estate=True)))
    templates.append(("compliance risk companies", make_label("risk_assessment")))
    for d in random.sample(districts, min(8, len(districts))):
        templates.append((f"RED PCB companies in {d}", make_label("risk_assessment", pcb="RED", district=d)))
        templates.append((f"risk companies outside estates in {d}", make_label("risk_assessment", outside_estate=True, district=d)))
    for sec_key, sec_words in SECTORS.items():
        w = random.choice(sec_words)
        templates.append((f"RED PCB {w} companies", make_label("risk_assessment", pcb="RED", sector=sec_key)))
    templates.append(("show supply chain risk LARGE companies depending on RED PCB tier suppliers", make_label("risk_assessment", pcb="RED", category="LARGE")))
    return templates

def gen_cluster():
    templates = []
    templates.append(("find production clusters", make_label("cluster_analysis")))
    templates.append(("supplier clusters 5+ units in same mandal", make_label("cluster_analysis")))
    templates.append(("find potential supplier clusters 5+ SMALL units in same mandal GREEN PCB", make_label("cluster_analysis", category="SMALL", pcb="GREEN")))
    templates.append(("which mandals have highest concentration of companies", make_label("cluster_analysis")))
    templates.append(("manufacturing hubs by mandal", make_label("cluster_analysis")))
    for sec_key, sec_words in SECTORS.items():
        w = random.choice(sec_words)
        templates.append((f"{w} clusters by mandal", make_label("cluster_analysis", sector=sec_key)))
    for d in random.sample(districts, min(6, len(districts))):
        templates.append((f"company clusters in {d}", make_label("cluster_analysis", district=d)))
    return templates

def gen_acquisition():
    templates = []
    templates.append(("find acquisition targets", make_label("acquisition")))
    templates.append(("MICRO companies with investment above 100 lakhs that could be acquisition targets", make_label("acquisition", category="MICRO", min_investment=100)))
    templates.append(("small companies worth acquiring", make_label("acquisition")))
    templates.append(("high investment MICRO companies", make_label("acquisition", category="MICRO")))
    templates.append(("find SMALL companies with high investment for acquisition", make_label("acquisition", category="SMALL")))
    for d in random.sample(districts, min(5, len(districts))):
        templates.append((f"acquisition targets in {d}", make_label("acquisition", district=d)))
    return templates

def gen_complex():
    """Complex multi-filter queries."""
    templates = []
    for sec_key, sec_words in SECTORS.items():
        w = random.choice(sec_words)
        d = random.choice(districts)
        for n in [5, 10]:
            templates.append((f"top {n} {w} companies in {d} by investment",
                make_label("ranking", sector=sec_key, district=d, limit=n)))
            templates.append((f"show {n} women-led {w} companies",
                make_label("diversity", women=True, sector=sec_key, limit=n)))
        templates.append((f"LARGE {w} companies in {d} outside estate",
            make_label("search", sector=sec_key, district=d, category="LARGE", outside_estate=True)))
        templates.append((f"RED PCB {w} companies with investment above 100 crore",
            make_label("risk_assessment", pcb="RED", sector=sec_key, min_investment=10000)))
    # Sort variations
    templates.append(("companies sorted by employment", make_label("search", sort_by="emp")))
    templates.append(("top companies by turnover", make_label("ranking", sort_by="turn")))
    templates.append(("MEGA companies ranked by employment", make_label("ranking", category="MEGA", sort_by="emp")))
    # Exclusion
    for d in random.sample(districts, min(4, len(districts))):
        templates.append((f"MEGA companies not in {d}", make_label("search", category="MEGA", exclude_district=d)))
        templates.append((f"all companies excluding {d}", make_label("search", exclude_district=d)))
    templates.append(("all companies excluding MICRO", make_label("search", exclude_category="MICRO")))
    templates.append(("companies not MICRO", make_label("search", exclude_category="MICRO")))
    return templates

def gen_natural_paraphrases():
    """More natural/conversational queries."""
    templates = []
    templates.append(("what are the biggest pharma companies", make_label("ranking", sector="health")))
    templates.append(("I want to see all textile mills", make_label("search", sector="textile")))
    templates.append(("give me food companies only 5", make_label("search", sector="food", limit=5)))
    templates.append(("construction firms in Rangareddy", make_label("search", sector="construction", district="Rangareddy")))
    templates.append(("how many cement companies are there", make_label("search", sector="construction")))
    templates.append(("who are the top employers in pharma", make_label("ranking", sector="health", sort_by="emp")))
    templates.append(("show me MEGA companies only", make_label("search", category="MEGA")))
    templates.append(("just 3 companies please", make_label("search", limit=3)))
    templates.append(("all companies in Medchal", make_label("search", district="Medchal")))
    templates.append(("health related companies", make_label("search", sector="health")))
    templates.append(("find all health related companies", make_label("search", sector="health")))
    templates.append(("medical companies in Hyderabad region", make_label("search", sector="health")))
    templates.append(("give top women led companies only 5", make_label("diversity", women=True, limit=5)))
    templates.append(("battery and energy companies", make_label("search", sector="energy")))
    templates.append(("which auto companies employ the most people", make_label("ranking", sector="auto", sort_by="emp")))
    templates.append(("ODOP companies with turnover above 500 lakhs rank by district", make_label("odop_analysis", odop=True, min_turnover=500)))
    templates.append(("MEGA in Sangareddy investment above 500 crore", make_label("search", category="MEGA", district="Sangareddy", min_investment=50000)))
    templates.append(("all the big companies", make_label("search", category="MEGA")))
    templates.append(("small scale industries", make_label("search", category="SMALL")))
    templates.append(("micro enterprises with good investment", make_label("acquisition", category="MICRO")))
    templates.append(("green pcb companies in estate", make_label("search", pcb="GREEN", in_estate=True)))
    templates.append(("dangerous polluting companies", make_label("risk_assessment", pcb="RED")))
    templates.append(("environmentally risky firms", make_label("risk_assessment")))
    templates.append(("find me 10 solar companies", make_label("search", sector="energy", limit=10)))
    templates.append(("plastic manufacturers", make_label("search", sector="chemical")))
    templates.append(("rice mills in Karimnagar", make_label("search", sector="food", district="Karimnagar")))
    templates.append(("top 5 women led pharma companies", make_label("diversity", women=True, sector="health", limit=5)))
    templates.append(("only show 3 results", make_label("search", limit=3)))
    templates.append(("bottom 10 companies by investment", make_label("ranking", limit=10, sort_by="inv_asc")))
    templates.append(("alphabetical list of companies", make_label("search", sort_by="name")))
    return templates

def gen_supply_demand():
    """Supply-demand gap analysis queries."""
    templates = []
    base_queries = [
        "show supply demand gap", "supply demand analysis", "where is demand high and supply low",
        "supply deficit districts", "which districts need more suppliers",
        "map supply and demand", "demand supply mismatch", "supply gap analysis",
        "where are buyers but no suppliers", "districts with supply shortage",
        "show companies where demand is high and supply is low",
        "supply chain gaps", "buyer supplier imbalance", "production gap analysis",
        "which areas have excess demand", "supply surplus districts",
        "where is supply high and demand low", "sourcing hub analysis",
        "find districts with more buyers than suppliers",
        "supply demand ratio by district", "demand hotspots",
        "which districts have supply surplus", "supply chain mapping",
        "map demand vs supply across districts", "identify supply gaps",
        "show demand concentration", "where do we need more small companies",
        "production capacity vs demand", "buyer concentration analysis",
        "find companies which can be linked to others based on supply or demand",
        "companies that can be linked based on supply",
        "link companies by supply and demand",
        "connect companies based on supply demand",
        "which companies can be connected as buyers and suppliers",
        "map company linkages", "show buyer supplier connections",
        "find linkable companies", "companies that could trade with each other",
        "potential supply chain connections", "match buyers with suppliers",
        "who can supply to whom", "link small to large companies",
        "find supply chain partners", "potential business connections",
        "companies that complement each other", "match demand with supply",
        "connect producers to buyers", "find trade partners",
        "map company relationships by supply demand",
        "which companies depend on each other",
        "show interdependent companies", "supply chain network",
        "find companies that could work together",
        "map how companies can be linked",
    ]
    for q in base_queries:
        templates.append((q, make_label("supply_demand")))
    for sec_key, sec_words in SECTORS.items():
        w = random.choice(sec_words)
        templates.append((f"{w} supply demand gap", make_label("supply_demand", sector=sec_key)))
        templates.append((f"supply deficit in {w} sector", make_label("supply_demand", sector=sec_key)))
        templates.append((f"where is {w} demand high", make_label("supply_demand", sector=sec_key)))
    for d in random.sample(districts, min(6, len(districts))):
        templates.append((f"supply demand analysis for {d}", make_label("supply_demand", district=d)))
    return templates

# ── BUILD DATASET ──
all_templates = []
all_templates.extend(gen_search())
all_templates.extend(gen_diversity())
all_templates.extend(gen_ranking())
all_templates.extend(gen_comparison())
all_templates.extend(gen_supplier_buyer())
all_templates.extend(gen_risk())
all_templates.extend(gen_cluster())
all_templates.extend(gen_acquisition())
all_templates.extend(gen_complex())
all_templates.extend(gen_natural_paraphrases())
all_templates.extend(gen_supply_demand())

# Augment with lowercase/uppercase/mixed variations
augmented = []
for query, label in all_templates:
    augmented.append((query, label))
    augmented.append((query.lower(), label))
    if random.random() < 0.3:
        augmented.append((query.upper(), label))
    if random.random() < 0.3:
        augmented.append((query.title(), label))

# Deduplicate by query text
seen = set()
final = []
for q, l in augmented:
    key = q.lower().strip()
    if key not in seen:
        seen.add(key)
        final.append({"query": q, "label": l})

random.shuffle(final)

# Split train/test
split = int(len(final) * 0.85)
train = final[:split]
test = final[split:]

os.makedirs(os.path.dirname(__file__) or '.', exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), 'train.json'), 'w') as f:
    json.dump(train, f, indent=2)
with open(os.path.join(os.path.dirname(__file__), 'test.json'), 'w') as f:
    json.dump(test, f, indent=2)

print(f"Generated {len(final)} total samples")
print(f"Train: {len(train)}, Test: {len(test)}")
print(f"Intent distribution:")
from collections import Counter
intent_counts = Counter(d["label"]["intent"] for d in final)
for intent, count in intent_counts.most_common():
    print(f"  {intent}: {count}")
print(f"\nSector distribution:")
sector_counts = Counter(d["label"].get("sector", "none") for d in final)
for sec, count in sector_counts.most_common(12):
    print(f"  {sec}: {count}")
print(f"\nSamples with limit: {sum(1 for d in final if 'limit' in d['label'])}")
print(f"Samples with sort_by: {sum(1 for d in final if 'sort_by' in d['label'])}")
print(f"Samples with district: {sum(1 for d in final if 'district' in d['label'])}")
