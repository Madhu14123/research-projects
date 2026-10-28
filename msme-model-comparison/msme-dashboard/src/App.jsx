import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import { districts as DISTRICTS_DATA, mandals as MANDALS_DATA, units as UNITS_DATA, totalMSMEs } from "./data.js";
import REL_DATA from "./relationships.json";
import { DIST_CENTROIDS } from "./map_data.js";
import { TG_DISTRICTS } from "./tg_districts.js";
import { TG_MANDALS } from "./tg_mandals.js";
import { TG_OUTLINE } from "./tg_outline.js";

// ????????? LOOKUP TABLES ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
const CATS = ["MICRO","SMALL","MEDIUM","LARGE","MEGA"];
const PCBS = ["GREEN","ORANGE","RED","WHITE"];
const SOCS = ["OC","OBC","SC","ST",""];
const CAT_COL = ["#378ADD","#1D9E75","#EF9F27","#D85A30","#7F77DD"];
const PCB_COL = ["#1D9E75","#EF9F27","#E24B4A","#888780"];
const PCB_LABEL = ["Low risk","Moderate risk","High risk","Minimal risk"];

// ????????? SUGGESTED QUERIES ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
const QUERIES = [
  "Find all pharma companies in Medchal",
  "Show health and medical companies across Telangana",
  "Find all MEGA companies in Sangareddy with investment above 500 crore",
  "List textile companies in Warangal",
  "Show RED PCB risk chemical companies outside industrial estates",
  "Find women-led food companies across all districts",
  "Which solar and energy companies have highest employment?",
  "Find potential supplier clusters: 5+ SMALL units in same mandal",
  "Compare Amara Raja and similar companies",
  "Which districts have the densest MEGA company concentration?",
  "Find auto and engineering companies in Rangareddy",
  "Show all construction companies with investment above 100 crore",
];

// ????????? SECTOR MAP ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
// queryKeys: matched against the user's query text (word-boundary safe)
// nameKeys: matched against company names in the dataset (substring match)
const SECTORS = {
  health: {
    label: "Health / Pharma",
    queryKeys: ["pharma","hospital","medical","health","clinic","drug","biotech","surgical","medicine","healthcare","wellness","diagnostic","laboratory","ayur","herbal"],
    nameKeys:  ["pharma","hospital","medical","health","clinic","drug","biotech","surgical","diagnos","laborator","ayur","herbal","capsule","tablet","syringe","vaccine","therap","oncol","formul","lifesci"],
  },
  textile: {
    label: "Textiles / Apparel",
    queryKeys: ["textile","garment","fabric","cotton","spinning","weaving","apparel","cloth","fashion"],
    nameKeys:  ["textile","garment","fabric","cotton","spinning","weaving","apparel","cloth","yarn","denim","knit","stitch","loom","dyeing","silk","jute","hosiery"],
  },
  food: {
    label: "Food / Agri",
    queryKeys: ["food","rice","flour","dairy","milk","poultry","bakery","spice","agri","agriculture","grain","meat","fish","aqua"],
    nameKeys:  ["food","rice","flour","dairy","milk","poultry","bakery","spice","oil mill","chicken","meat","sugar","grain","cattle","feed","fish","aqua","prawn","edible","masala","biscuit","noodle","snack","beverage","water plant","ice cream","paraboil","parboil"],
  },
  construction: {
    label: "Construction / Building",
    queryKeys: ["cement","brick","concrete","steel","construction","granite","marble","tiles","building","infrastructure","real estate"],
    nameKeys:  ["cement","brick","concrete","steel","construction","granite","marble","sand ","stone","tiles","readymix","rmc ","fly ash","aac block","lime","gypsum","roofing","plumb","infra"],
  },
  auto: {
    label: "Auto / Engineering",
    queryKeys: ["auto","automobile","automotive","engineering","motor","vehicle","battery","machiner","casting","fabricat","tractor"],
    nameKeys:  ["auto","engine","motor","vehicle","battery","machiner","casting","forg","weld","fabricat","cnc","lathe","precision","gear","bearing","piston","axle","chassis","tractor","two wheeler"],
  },
  energy: {
    label: "Energy / Power",
    queryKeys: ["solar","power","energy","electric","wind","biomass","renewable","turbine","transformer"],
    nameKeys:  ["solar","power plant","energy","wind ","biomass","generat","turbine","transform","inverter","lithium","cell tech"],
  },
  chemical: {
    label: "Chemical / Plastics",
    queryKeys: ["chemical","paint","plastic","polymer","rubber","fertiliz","pesticide","petrochemical"],
    nameKeys:  ["chemical","paint","plastic","polymer","rubber","fertiliz","pesticide","resin","pigment","adhesive","solvent","acid","coating","ink ","pvc","hdpe","fibre glass"],
  },
  it: {
    label: "IT / Electronics",
    queryKeys: ["software","electronics","semiconductor","chip","circuit","computer","it company","tech company","iot","sensor"],
    nameKeys:  ["software","tech park","data center","semicon","chip","electron","circuit","iot","sensor","display","computer","server","infosys","wipro"],
  },
  packaging: {
    label: "Packaging / Paper",
    queryKeys: ["packaging","paper","carton","corrugat","printing"],
    nameKeys:  ["packag","paper","carton","corrugat","box ","printing","label","pouch","wrapper","cardboard","pulp"],
  },
  furniture: {
    label: "Furniture / Wood",
    queryKeys: ["furniture","wood","plywood","timber","laminate"],
    nameKeys:  ["furniture","wood","plywood","timber","laminate","veneer","mdf","particl","carpent"],
  },
};

function detectSector(q) {
  for (const [key, sec] of Object.entries(SECTORS)) {
    if (sec.queryKeys.some(k => {
      const re = new RegExp("\\b" + k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + "\\b", "i");
      return re.test(q);
    })) return key;
  }
  return null;
}

function filterBySector(unitsList, sectorKey) {
  const keys = SECTORS[sectorKey].nameKeys;
  return unitsList.filter(u => {
    const name = u[1].toLowerCase();
    return keys.some(k => name.includes(k));
  });
}

function getSec(u) {
  const nm = u[1].toLowerCase();
  for (const [sk, sec] of Object.entries(SECTORS)) {
    if (sec.nameKeys.some(k => nm.includes(k))) return sk;
  }
  return null;
}

// ????????? QUERY PARSER ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function parseQuery(text, units, districts, mandals) {
  const q = text.toLowerCase().trim();

  // ?????? Result limit: "top 5", "only 5", "first 10", "5 companies", "limit 20"
  let limit = null;
  const limMatch = q.match(/(?:top|only|first|show|give|just|limit)\s+(\d+)/);
  if (limMatch) limit = parseInt(limMatch[1]);
  const limMatch2 = q.match(/(\d+)\s+(?:companies|results|units|firms|entries)/);
  if (!limit && limMatch2) limit = parseInt(limMatch2[1]);
  if (limit && limit > 200) limit = 200;

  // ?????? Sort preference
  let sortBy = "inv";
  if (/by\s+employ|by\s+worker|by\s+emp|highest\s+employ|most\s+employ|most\s+worker/.test(q)) sortBy = "emp";
  else if (/by\s+turnover|highest\s+turnover|most\s+turnover|by\s+revenue/.test(q)) sortBy = "turn";
  else if (/by\s+name|alphabetic/.test(q)) sortBy = "name";
  else if (/smallest|lowest|least\s+invest|bottom/.test(q)) sortBy = "inv_asc";

  // ?????? Intent detection (order matters - more specific first)
  let intent = "general";
  if (/supplier|supply to|who supplies|vendor|feeder|input provider/.test(q)) intent = "supplier_search";
  else if (/buyer|customer|who buys|demand|purchase|anchor buyer/.test(q)) intent = "buyer_search";
  else if (/risk|red pcb|outside estate|pollution|hazard|compliance|danger/.test(q)) intent = "risk_assessment";
  else if (/cluster|concentration|dense|hub|5\+|corridor|hotspot/.test(q)) intent = "cluster_analysis";
  else if (/compare|vs\b|versus|similar|benchmark|like\b/.test(q)) intent = "comparison";
  else if (/women[\s-]*led|female[\s-]*led|women\s+own|female\s+promot|woman/.test(q)) intent = "diversity";
  else if (/odop|one district one product/.test(q)) intent = "odop_analysis";
  else if (/acquisition|takeover|acquire|invest in|buyout/.test(q)) intent = "acquisition";
  else if (/\brank\b|densest|which district/.test(q)) intent = "ranking";
  else if (/\btop\s+\d|biggest|largest|highest/.test(q)) intent = "ranking";
  else if (/list|find|show|all\b|which|where|give|get/.test(q)) intent = "search";

  const filters = {};

  // ?????? Sector (uses word-boundary matching on query text)
  const sector = detectSector(q);
  if (sector) filters.sector = sector;

  // ?????? District
  for (let i = 0; i < districts.length; i++) {
    if (districts[i] && districts[i] !== "Unknown") {
      const dLow = districts[i].toLowerCase();
      const re = new RegExp("\\b" + dLow.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + "\\b");
      if (re.test(q)) { filters.district = districts[i]; break; }
    }
  }

  // ?????? Category (word boundary)
  for (const c of CATS) {
    const re = new RegExp("\\b" + c.toLowerCase() + "\\b");
    if (re.test(q)) { filters.category = c; break; }
  }
  // Multi-category: "LARGE or MEDIUM", "LARGE and MEDIUM"
  if (/large.{1,5}medium|medium.{1,5}large/i.test(q)) {
    filters.categories = ["LARGE", "MEDIUM"];
    delete filters.category;
  }
  if (/small.{1,5}micro|micro.{1,5}small/i.test(q)) {
    filters.categories = ["SMALL", "MICRO"];
    delete filters.category;
  }

  // ?????? PCB tier
  if (/\bred\b.*pcb|\bred\b.*tier|\bred\b.*risk|\bred\b.*pollution/.test(q)) filters.pcb = "RED";
  else if (/\bgreen\b.*pcb|\bgreen\b.*tier/.test(q)) filters.pcb = "GREEN";
  else if (/\borange\b.*pcb|\borange\b.*tier/.test(q)) filters.pcb = "ORANGE";
  else if (/\bwhite\b.*pcb|\bwhite\b.*tier/.test(q)) filters.pcb = "WHITE";

  // ?????? Investment thresholds
  const crMatch = q.match(/(\d+)\s*(?:crore|cr\b)/);
  if (crMatch) filters.minInvestment = parseInt(crMatch[1]) * 100;
  const lMatch = q.match(/(?:above|over|more than|greater than|>=?|minimum|atleast|at least)?\s*(\d+)\s*(?:lakh|lakhs|l\b)/);
  if (lMatch && !crMatch) filters.minInvestment = parseInt(lMatch[1]);

  // ?????? Turnover thresholds
  const turnCrMatch = q.match(/turnover.*?(\d+)\s*(?:crore|cr\b)/);
  if (turnCrMatch) filters.minTurnover = parseInt(turnCrMatch[1]) * 100;
  const turnLMatch = q.match(/turnover.*?(\d+)\s*(?:lakh|lakhs|l\b)/);
  if (turnLMatch && !turnCrMatch) filters.minTurnover = parseInt(turnLMatch[1]);

  // ?????? Employment threshold
  const empMatch = q.match(/(\d+)\+?\s*(?:employ|worker|people|staff|emp\b)/);
  if (empMatch) filters.minEmployment = parseInt(empMatch[1]);

  // ?????? Boolean flags (word-boundary safe)
  if (/\bwomen\b|\bfemale\b/.test(q)) filters.women = true;
  if (/\bodop\b/.test(q)) filters.odop = true;
  if (/outside.*estate|not.*estate|dispersed/.test(q)) filters.outsideEstate = true;
  if (/\bin\s+.*estate|\binside.*estate/.test(q) && !/outside/.test(q)) filters.inEstate = true;

  // ?????? Exclusions: "not in Medchal", "excluding MICRO"
  let exclude = {};
  const exclDistMatch = q.match(/(?:not in|excluding|except|outside)\s+([a-z]+)/);
  if (exclDistMatch) {
    const ed = districts.find(d => d && d.toLowerCase() === exclDistMatch[1]);
    if (ed) exclude.district = ed;
  }
  if (/excluding\s+micro|not\s+micro|no\s+micro/.test(q)) exclude.category = "MICRO";

  // ?????? Target company
  let targetCompany = null;
  const bigNames = ["amara raja","dr. reddy","dr.reddy","aurobindo","kitex","ramagundam","hetero","natco","granules","laurus","msn lab","sagar cement","orient cement","mahindra","mrf","hpcl","itc limited","welspun","bharath electron","sanofi","unique biotech"];
  for (const kn of bigNames) {
    if (q.includes(kn)) { targetCompany = units.find(u => u[1].toLowerCase().includes(kn)); break; }
  }
  if (!targetCompany) {
    for (const u of units) {
      const nm = u[1].toLowerCase();
      if (nm.length > 6 && q.includes(nm)) { targetCompany = u; break; }
    }
  }

  return { intent, filters, exclude, targetCompany, limit, sortBy };
}

// ????????? SMART CONTEXT BUILDER ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function buildSmartContext(parsed, units, districts, mandals) {
  const { intent, filters, targetCompany } = parsed;
  let pool = [...units];

  if (filters.district) pool = pool.filter(u => districts[u[2]] === filters.district);
  if (filters.category) pool = pool.filter(u => CATS[u[4]] === filters.category);
  if (filters.pcb) pool = pool.filter(u => PCBS[u[5]] === filters.pcb);
  if (filters.women) pool = pool.filter(u => u[11] === 1);
  if (filters.odop) pool = pool.filter(u => u[12] === 1);
  if (filters.outsideEstate) pool = pool.filter(u => u[10] === 0);
  if (filters.inEstate) pool = pool.filter(u => u[10] === 1);
  if (filters.minInvestment) pool = pool.filter(u => u[7] >= filters.minInvestment);

  pool.sort((a, b) => b[7] - a[7]);

  let chain = [];
  if (targetCompany) {
    const sameDistrict = units.filter(u => u[2] === targetCompany[2] && u[0] !== targetCompany[0]);
    const suppliers = sameDistrict.filter(u => u[4] < targetCompany[4]).sort((a,b) => b[7]-a[7]).slice(0, 15);
    const buyers = sameDistrict.filter(u => u[4] > targetCompany[4]).sort((a,b) => b[7]-a[7]).slice(0, 10);
    const peers = sameDistrict.filter(u => u[4] === targetCompany[4]).sort((a,b) => b[7]-a[7]).slice(0, 10);
    const mandalNeighbors = units.filter(u => u[3] === targetCompany[3] && u[0] !== targetCompany[0]).sort((a,b) => b[7]-a[7]).slice(0, 10);
    chain = [...suppliers, ...buyers, ...peers, ...mandalNeighbors];
    const seen = new Set(); chain = chain.filter(u => seen.has(u[0]) ? false : (seen.add(u[0]), true));
  }

  if (intent === "cluster_analysis") {
    const mandalCounts = {};
    pool.forEach(u => { const m = u[3]; mandalCounts[m] = (mandalCounts[m]||0) + 1; });
    const hotMandals = Object.entries(mandalCounts).filter(([,c]) => c >= 5).map(([m]) => parseInt(m));
    if (hotMandals.length) pool = pool.filter(u => hotMandals.includes(u[3]));
  }

  if (intent === "risk_assessment" && !filters.pcb) {
    pool = pool.filter(u => u[5] === 2);
  }

  const toObj = u => ({
    id:u[0], name:u[1], district:districts[u[2]]||"", mandal:mandals[u[3]]||"",
    category:CATS[u[4]], pcb:PCBS[u[5]], social:SOCS[u[6]],
    investment_lakhs:u[7], employment:u[8], turnover_lakhs:u[9],
    in_estate:u[10]===1, women_led:u[11]===1, odop:u[12]===1,
  });

  return {
    primary: pool.slice(0, 200).map(toObj),
    supplyChain: chain.slice(0, 50).map(toObj),
    targetCompany: targetCompany ? toObj(targetCompany) : null,
    totalMatching: pool.length,
    intent,
    filters,
  };
}

// ????????? MATCHING ENGINE ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function findSimilarCompanies(target, units, n = 10) {
  if (!target) return [];
  const maxInv = Math.max(...units.map(u => u[7])) || 1;
  const maxEmp = Math.max(...units.map(u => u[8])) || 1;
  const maxTurn = Math.max(...units.map(u => u[9])) || 1;
  const tI = target[7]/maxInv, tE = target[8]/maxEmp, tT = target[9]/maxTurn;

  return units.filter(u => u[0] !== target[0]).map(u => {
    let d = Math.sqrt(
      Math.pow((u[7]/maxInv - tI), 2) * 3 +
      Math.pow((u[8]/maxEmp - tE), 2) * 2 +
      Math.pow((u[9]/maxTurn - tT), 2)
    );
    if (u[4] === target[4]) d *= 0.7;
    if (u[2] === target[2]) d *= 0.8;
    if (u[5] === target[5]) d *= 0.9;
    return { unit: u, score: 1 / (1 + d) };
  }).sort((a, b) => b.score - a.score).slice(0, n);
}

// ????????? SUPPLY CHAIN BUILDER ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function buildSupplyNetwork(target, units, districts, mandals) {
  if (!target) return { nodes: [], links: [] };
  const nodes = [{ id: target[0], name: target[1], catIdx: target[4], pcbIdx: target[5], investment: target[7], role: "target" }];
  const links = [];
  const same = units.filter(u => u[2] === target[2] && u[0] !== target[0]);

  const suppliers = same.filter(u => u[4] < target[4]).sort((a,b) => b[7]-a[7]).slice(0, 8);
  suppliers.forEach(u => {
    nodes.push({ id: u[0], name: u[1], catIdx: u[4], pcbIdx: u[5], investment: u[7], role: "supplier" });
    links.push({ source: u[0], target: target[0], type: "supplier" });
  });

  const buyers = same.filter(u => u[4] > target[4]).sort((a,b) => b[7]-a[7]).slice(0, 5);
  buyers.forEach(u => {
    nodes.push({ id: u[0], name: u[1], catIdx: u[4], pcbIdx: u[5], investment: u[7], role: "buyer" });
    links.push({ source: target[0], target: u[0], type: "buyer" });
  });

  const mandalPeers = units.filter(u => u[3] === target[3] && u[0] !== target[0]
    && !nodes.find(n => n.id === u[0])).sort((a,b) => b[7]-a[7]).slice(0, 6);
  mandalPeers.forEach(u => {
    nodes.push({ id: u[0], name: u[1], catIdx: u[4], pcbIdx: u[5], investment: u[7], role: "peer" });
    links.push({ source: target[0], target: u[0], type: "peer" });
  });

  const riskLinks = suppliers.filter(u => u[5] === 2 && u[10] === 0);
  riskLinks.forEach(u => {
    links.push({ source: u[0], target: target[0], type: "risk" });
  });

  return { nodes, links };
}

// ????????? FORCE LAYOUT ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function computeForceLayout(nodes, links, W = 600, H = 450) {
  if (!nodes.length) return [];
  const pos = nodes.map((n, i) => ({
    ...n,
    x: W/2 + (W/3) * Math.cos(2 * Math.PI * i / nodes.length),
    y: H/2 + (H/3) * Math.sin(2 * Math.PI * i / nodes.length),
    vx: 0, vy: 0,
  }));
  if (pos[0]) { pos[0].x = W/2; pos[0].y = H/2; }

  for (let iter = 0; iter < 120; iter++) {
    for (let i = 0; i < pos.length; i++) {
      for (let j = i+1; j < pos.length; j++) {
        const dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y;
        const dist = Math.max(Math.sqrt(dx*dx + dy*dy), 1);
        const f = 4000 / (dist * dist);
        pos[i].vx -= (dx/dist)*f; pos[i].vy -= (dy/dist)*f;
        pos[j].vx += (dx/dist)*f; pos[j].vy += (dy/dist)*f;
      }
    }
    for (const l of links) {
      const s = pos.find(p => p.id === l.source), t = pos.find(p => p.id === l.target);
      if (!s || !t) continue;
      const dx = t.x - s.x, dy = t.y - s.y;
      const dist = Math.max(Math.sqrt(dx*dx + dy*dy), 1);
      const f = (dist - 90) * 0.012;
      s.vx += (dx/dist)*f; s.vy += (dy/dist)*f;
      t.vx -= (dx/dist)*f; t.vy -= (dy/dist)*f;
    }
    for (const p of pos) {
      p.vx += (W/2 - p.x) * 0.001; p.vy += (H/2 - p.y) * 0.001;
      p.x += p.vx * 0.5; p.y += p.vy * 0.5;
      p.vx *= 0.78; p.vy *= 0.78;
      p.x = Math.max(40, Math.min(W-40, p.x));
      p.y = Math.max(40, Math.min(H-40, p.y));
    }
  }
  return pos;
}

const fmt = (n, unit="Rs.") => {
  if (!n || n===0) return "--";
  if (n>=10000000) return `${unit}${(n/10000000).toFixed(1)}Cr`;
  if (n>=100000) return `${unit}${(n/100000).toFixed(1)}L`;
  if (n>=1000) return `${unit}${(n/1000).toFixed(1)}K`;
  return `${unit}${n}`;
};
const fmtE = (n) => n >= 1000 ? `${(n/1000).toFixed(1)}K` : String(n||"--");

export default function App() {
  const units = UNITS_DATA;
  const districts = DISTRICTS_DATA;
  const mandals = MANDALS_DATA;
  const loading = false;
  const [tab, setTab]           = useState("search");

  // Search state
  const [query, setQuery]       = useState("");
  const [filterDist, setFilterDist] = useState("ALL");
  const [filterCat, setFilterCat]   = useState("ALL");
  const [filterPcb, setFilterPcb]   = useState("ALL");
  const [filterWomen, setFilterWomen] = useState(false);
  const [filterOdop, setFilterOdop]   = useState(false);
  const [filterEstate, setFilterEstate] = useState(false);
  const [filterMinInv, setFilterMinInv] = useState("");
  const [filterMaxInv, setFilterMaxInv] = useState("");
  const [filterMinEmp, setFilterMinEmp] = useState("");
  const [filterMandal, setFilterMandal] = useState("ALL");
  const [sortBy, setSortBy]     = useState("inv");
  const [page, setPage]         = useState(0);
  const PAGE_SIZE = 25;

  // Company detail
  const [selected, setSelected] = useState(null);

  // Analytics filters (independent from search)
  const [aRisk, setARisk] = useState("all");           // risk profile
  const [aIntensity, setAIntensity] = useState("all"); // capital vs labor intensity
  const [aScale, setAScale] = useState("all");         // scale potential
  const [aDensity, setADensity] = useState("all");     // cluster density
  const [aEmployer, setAEmployer] = useState("all");   // employer size
  const [aDist, setADist] = useState("ALL");           // district (for analytics)

  // Pre-compute mandal density lookup once
  const mandalDensity = useMemo(() => {
    const counts = {};
    units.forEach(u => { const m = u[3]; counts[m] = (counts[m]||0)+1; });
    return counts;
  }, []);

  // AI query
  const [aiQuery, setAiQuery]   = useState("");
  const [aiResult, setAiResult] = useState(null);

  // Network graph
  const [networkTarget, setNetworkTarget] = useState(null);
  const [networkData, setNetworkData]     = useState({ nodes: [], links: [] });
  const [networkPositions, setNetworkPositions] = useState([]);
  const [similarResults, setSimilarResults]     = useState([]);

  // Map state (abstract explorer)
  const [selectedDist, setSelectedDist] = useState(null);
  const [selectedLink, setSelectedLink] = useState(null);
  const [selectedNetLink, setSelectedNetLink] = useState(null);
  const [exploreHistory, setExploreHistory] = useState([]);

  // Geographic supply map ??? zoom-based cluster -> company view
  const [mapView, setMapView] = useState({ minLat: 15.65, maxLat: 20.05, minLon: 77.05, maxLon: 81.55 });
  const [mapSelCo, setMapSelCo] = useState(null);
  const [mapSelLk, setMapSelLk] = useState(null);
  const [mapCluster, setMapCluster] = useState(null); // cluster detail panel (all companies)

  // All companies ? those with valid GPS use their own coords,
  // those without fall back to their district centroid so all 59,862 appear on the map.
  const geoUnits = useMemo(() => {
    const TG_LAT=[15.75,19.95], TG_LON=[77.15,81.65];
    return units.map(u => {
      if (u[13]>TG_LAT[0] && u[13]<TG_LAT[1] && u[14]>TG_LON[0] && u[14]<TG_LON[1]) return u;
      const cent = DIST_CENTROIDS[u[2]];
      if (!cent) return null;
      const uCopy = u.slice();
      uCopy[13] = cent[0]; uCopy[14] = cent[1];
      return uCopy;
    }).filter(Boolean);
  }, []);

  // Per-company link lookup (id -> links array) for side panel
  const coLinks = useMemo(() => {
    const m = {};
    (REL_DATA.links || []).forEach(l => {
      if (!m[l.source]) m[l.source] = [];
      if (!m[l.target]) m[l.target] = [];
      m[l.source].push(l);
      m[l.target].push(l);
    });
    return m;
  }, []);

  const exploreTo = useCallback((item) => {
    setExploreHistory(h => selectedLink ? [...h, selectedLink] : h);
    setSelectedLink(item);
  }, [selectedLink]);

  const exploreBack = useCallback(() => {
    setExploreHistory(h => {
      if (!h.length) { setSelectedLink(null); return []; }
      setSelectedLink(h[h.length - 1]);
      return h.slice(0, -1);
    });
  }, []);

  // ????????? FILTERED + SORTED UNITS ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
  const filtered = useMemo(() => {
    if (!units.length) return [];
    const q = query.toLowerCase().trim();
    return units.filter(u => {
      if (q && !u[1].toLowerCase().includes(q) && !(mandals[u[3]]||"").toLowerCase().includes(q)) return false;
      if (filterDist !== "ALL" && districts[u[2]] !== filterDist) return false;
      if (filterCat !== "ALL" && CATS[u[4]] !== filterCat) return false;
      if (filterPcb !== "ALL" && PCBS[u[5]] !== filterPcb) return false;
      if (filterWomen && !u[11]) return false;
      if (filterOdop && !u[12]) return false;
      if (filterEstate && !u[10]) return false;
      if (filterMinInv && u[7] < parseFloat(filterMinInv)) return false;
      if (filterMaxInv && u[7] > parseFloat(filterMaxInv)) return false;
      if (filterMinEmp && u[8] < parseInt(filterMinEmp)) return false;
      if (filterMandal !== "ALL" && mandals[u[3]] !== filterMandal) return false;
      return true;
    }).sort((a,b) => {
      if (sortBy==="inv") return b[7]-a[7];
      if (sortBy==="emp") return b[8]-a[8];
      if (sortBy==="turn") return b[9]-a[9];
      if (sortBy==="name") return a[1].localeCompare(b[1]);
      return 0;
    });
  }, [units, query, filterDist, filterCat, filterPcb, filterWomen, filterOdop, filterEstate, filterMinInv, filterMaxInv, filterMinEmp, filterMandal, sortBy, districts, mandals]);

  const paginated = useMemo(() => filtered.slice(page*PAGE_SIZE, (page+1)*PAGE_SIZE), [filtered, page]);
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);

  // ????????? COMPANY DETAIL ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
  const getCompanyDetail = useCallback((u) => {
    // Find similar companies: same district, same PCB tier
    const similar = units
      .filter(x => x[2]===u[2] && x[4]===u[4] && x[0]!==u[0])
      .sort((a,b) => b[7]-a[7]).slice(0,5);
    // Find nearby mandal companies
    const mandal_peers = units
      .filter(x => x[3]===u[3] && x[0]!==u[0])
      .sort((a,b) => b[7]-a[7]).slice(0,5);
    // Potential suppliers (smaller units, same district)
    const suppliers = units
      .filter(x => x[2]===u[2] && x[4]<u[4] && x[0]!==u[0])
      .sort((a,b) => b[7]-a[7]).slice(0,5);
    return { similar, mandal_peers, suppliers };
  }, [units]);

  // ????????? NETWORK BUILDER ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
  const openNetwork = useCallback((u) => {
    setNetworkTarget(u);
    setSelectedNetLink(null);
    const net = buildSupplyNetwork(u, units, districts, mandals);
    setNetworkData(net);
    setNetworkPositions(computeForceLayout(net.nodes, net.links));
    setSimilarResults(findSimilarCompanies(u, units, 12));
    setTab("network");
  }, [units, districts, mandals]);

  // ????????? QUERY ENGINE ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
  // Applies parsed filters to data and generates results
  const executeQuery = useCallback((qText, p) => {
    const intent = p.intent || "general";
    const filters = {};
    if (p.sector) filters.sector = p.sector;
    if (p.district) filters.district = p.district;
    if (p.category) filters.category = p.category;
    if (p.categories) filters.categories = p.categories;
    if (p.pcb) filters.pcb = p.pcb;
    if (p.women) filters.women = true;
    if (p.odop) filters.odop = true;
    if (p.outside_estate) filters.outsideEstate = true;
    if (p.in_estate) filters.inEstate = true;
    if (p.min_investment) filters.minInvestment = p.min_investment;
    if (p.min_turnover) filters.minTurnover = p.min_turnover;
    if (p.min_employment) filters.minEmployment = p.min_employment;
    const exclude = {};
    if (p.exclude_district) exclude.district = p.exclude_district;
    if (p.exclude_category) exclude.category = p.exclude_category;
    const maxResults = p.limit || 30;
    const qSortBy = p.sort_by || "inv";

    // Find target company if specified
    let targetCompany = null;
    if (p.target_company) {
      const tc = p.target_company.toLowerCase();
      targetCompany = units.find(u => u[1].toLowerCase().includes(tc));
    }

    // Apply filters to get matching pool
    let pool = [...units];
    if (filters.sector) pool = filterBySector(pool, filters.sector);
    if (filters.district) pool = pool.filter(u => districts[u[2]] === filters.district);
    if (filters.category) pool = pool.filter(u => CATS[u[4]] === filters.category);
    if (filters.categories) pool = pool.filter(u => filters.categories.includes(CATS[u[4]]));
    if (filters.pcb) pool = pool.filter(u => PCBS[u[5]] === filters.pcb);
    if (filters.women) pool = pool.filter(u => u[11] === 1);
    if (filters.odop) pool = pool.filter(u => u[12] === 1);
    if (filters.outsideEstate) pool = pool.filter(u => u[10] === 0);
    if (filters.inEstate) pool = pool.filter(u => u[10] === 1);
    if (filters.minInvestment) pool = pool.filter(u => u[7] >= filters.minInvestment);
    if (filters.minTurnover) pool = pool.filter(u => u[9] >= filters.minTurnover);
    if (filters.minEmployment) pool = pool.filter(u => u[8] >= filters.minEmployment);
    if (exclude.district) pool = pool.filter(u => districts[u[2]] !== exclude.district);
    if (exclude.category) pool = pool.filter(u => CATS[u[4]] !== exclude.category);
    if (qSortBy === "emp") pool.sort((a,b) => b[8]-a[8]);
    else if (qSortBy === "turn") pool.sort((a,b) => b[9]-a[9]);
    else if (qSortBy === "name") pool.sort((a,b) => a[1].localeCompare(b[1]));
    else if (qSortBy === "inv_asc") pool.sort((a,b) => a[7]-b[7]);
    else pool.sort((a, b) => b[7] - a[7]);

    const toCompany = u => ({
      id: u[0], name: u[1], district: districts[u[2]]||"",
      mandal: mandals[u[3]]||"", category: CATS[u[4]],
      investment: u[7], employment: u[8], turnover: u[9],
      note: [u[11]===1?"Women-led":"", u[12]===1?"ODOP":"", u[10]===1?"In estate":"Outside estate", PCBS[u[5]]+" PCB"].filter(Boolean).join(" . "),
    });

    let prose = "";
    let companies = [];
    let links = [];
    let insight = "";
    let recommendation = "";
    let groupData = null;

    // ?????? INTENT HANDLERS ??????

    if (intent === "supply_demand") {
      // Supply-demand gap analysis by district
      // Demand = MEGA+LARGE (buyers needing inputs), Supply = SMALL+MICRO (producers)
      const analysis = {};
      const srcPool = filters.sector ? filterBySector(units, filters.sector) : units;
      srcPool.forEach(u => {
        const d = districts[u[2]] || "Unknown";
        if (!analysis[d]) analysis[d] = { demand: 0, supply: 0, demandInv: 0, supplyInv: 0, demandEmp: 0, supplyEmp: 0, demandUnits: [], supplyUnits: [] };
        if (u[4] >= 3) { // LARGE, MEGA = demand side
          analysis[d].demand++;
          analysis[d].demandInv += u[7];
          analysis[d].demandEmp += u[8];
          analysis[d].demandUnits.push(u);
        } else { // MICRO, SMALL, MEDIUM = supply side
          analysis[d].supply++;
          analysis[d].supplyInv += u[7];
          analysis[d].supplyEmp += u[8];
          analysis[d].supplyUnits.push(u);
        }
      });

      // Compute gap ratio: demand / supply (>1 = supply deficit, <1 = supply surplus)
      const gaps = Object.entries(analysis)
        .filter(([,v]) => v.demand > 0 || v.supply > 0)
        .map(([d, v]) => ({
          district: d,
          demand: v.demand,
          supply: v.supply,
          ratio: v.supply > 0 ? (v.demand / v.supply).toFixed(2) : "no supply",
          gap: v.demand - v.supply,
          demandInv: v.demandInv,
          supplyInv: v.supplyInv,
          demandUnits: v.demandUnits,
          supplyUnits: v.supplyUnits,
        }))
        .sort((a, b) => (typeof b.ratio === 'string' ? 999 : parseFloat(b.ratio)) - (typeof a.ratio === 'string' ? 999 : parseFloat(a.ratio)));

      const supplyDeficit = gaps.filter(g => typeof g.ratio !== 'string' && parseFloat(g.ratio) > 0.3);
      const supplySurplus = gaps.filter(g => typeof g.ratio !== 'string' && parseFloat(g.ratio) <= 0.1);
      const sectorLabel = filters.sector ? SECTORS[filters.sector].label : "all sectors";

      // Show top demand-heavy districts' anchor companies
      const topDeficit = supplyDeficit.slice(0, 5);
      const anchorCompanies = topDeficit.flatMap(g => g.demandUnits.sort((a,b) => b[7]-a[7]).slice(0, 3));
      companies = anchorCompanies.slice(0, maxResults).map(toCompany);

      // Supply chain links: connect biggest buyer to biggest supplier in same district
      topDeficit.forEach(g => {
        const topBuyer = g.demandUnits.sort((a,b) => b[7]-a[7])[0];
        const topSupplier = g.supplyUnits.sort((a,b) => b[7]-a[7])[0];
        if (topBuyer && topSupplier) {
          links.push({ from: topSupplier[1], to: topBuyer[1], type: "supplier" });
        }
      });

      prose = `Supply-demand gap analysis for ${sectorLabel}:\n\nDemand = MEGA + LARGE companies (buyers needing inputs)\nSupply = MICRO + SMALL + MEDIUM companies (producers)\n\n` +
        `HIGH DEMAND (supply deficit - need more suppliers):\n` +
        supplyDeficit.slice(0, 8).map(g =>
          `  ${g.district}: ${g.demand} buyers vs ${g.supply} suppliers (ratio ${g.ratio}) - demand inv ${fmt(g.demandInv)}`
        ).join("\n") +
        `\n\nHIGH SUPPLY (surplus - sourcing hubs):\n` +
        supplySurplus.slice(0, 8).map(g =>
          `  ${g.district}: ${g.supply} suppliers vs ${g.demand} buyers - supply inv ${fmt(g.supplyInv)}`
        ).join("\n");

      insight = supplyDeficit[0]
        ? `${supplyDeficit[0].district} has the biggest supply gap: ${supplyDeficit[0].demand} buyers but only ${supplyDeficit[0].supply} suppliers`
        : "Supply and demand are relatively balanced across districts";
      recommendation = supplySurplus[0] && supplyDeficit[0]
        ? `Connect ${supplySurplus[0].district} suppliers (${supplySurplus[0].supply} units) to ${supplyDeficit[0].district} buyers (${supplyDeficit[0].demand} units)`
        : "Consider incentivizing SMALL/MICRO units in demand-heavy districts";

      groupData = {
        type: "district_bar",
        data: gaps.slice(0, 12).map(g => ({
          label: g.district,
          value: g.demand,
          sub: `S:${g.supply}`,
        })),
      };
    }
    else if (intent === "comparison" && targetCompany) {
      const similar = findSimilarCompanies(targetCompany, units, 10);
      companies = [toCompany(targetCompany), ...similar.map(s => toCompany(s.unit))];
      const tc = targetCompany;
      const avgInv = similar.reduce((s,x) => s+x.unit[7], 0) / similar.length;
      const avgEmp = similar.reduce((s,x) => s+x.unit[8], 0) / similar.length;
      prose = `Comparing ${tc[1]} (${CATS[tc[4]]}, ${districts[tc[2]]}) against ${similar.length} most similar companies.\n\n${tc[1]}: Investment ${fmt(tc[7])}, Employment ${fmtE(tc[8])}, Turnover ${fmt(tc[9])}, PCB ${PCBS[tc[5]]}\nPeer average: Investment ${fmt(Math.round(avgInv))}, Employment ${fmtE(Math.round(avgEmp))}\n\n${tc[7] > avgInv ? tc[1]+" invests "+(tc[7]/avgInv).toFixed(1)+"x more than peers." : tc[1]+" invests less than peer average."}`;
      insight = tc[7] > avgInv*2 ? `${tc[1]} is a dominant player - ${(tc[7]/avgInv).toFixed(1)}x peer investment` : `${tc[1]} is comparable to peers in scale`;
      links = similar.slice(0,5).map(s => ({
        from: tc[1], to: s.unit[1],
        type: s.unit[4] < tc[4] ? "supplier" : s.unit[4] > tc[4] ? "buyer" : "peer",
      }));
    }
    else if ((intent === "supplier_search" || intent === "buyer_search") && targetCompany) {
      const target = targetCompany;
      const sameDistrict = units.filter(u => u[2] === target[2] && u[0] !== target[0]);
      const suppliers = sameDistrict.filter(u => u[4] < target[4]).sort((a,b) => b[7]-a[7]).slice(0,20);
      const buyers = sameDistrict.filter(u => u[4] > target[4]).sort((a,b) => b[7]-a[7]).slice(0,10);
      const list = intent === "supplier_search" ? suppliers : buyers;
      companies = [toCompany(target), ...list.map(toCompany)];
      links = list.slice(0,8).map(u => ({
        from: intent==="supplier_search" ? u[1] : target[1],
        to: intent==="supplier_search" ? target[1] : u[1],
        type: intent==="supplier_search" ? "supplier" : "buyer",
      }));
      const riskCount = list.filter(u => u[5] === 2 && u[10] === 0).length;
      prose = `Found ${list.length} potential ${intent==="supplier_search"?"suppliers":"buyers"} for ${target[1]} in ${districts[target[2]]}.\n\nTarget: ${target[1]} (${CATS[target[4]]}, inv ${fmt(target[7])}, emp ${fmtE(target[8])})\n${intent==="supplier_search"?"Suppliers":"Buyers"}: ${list.length} companies, total investment ${fmt(list.reduce((s,u)=>s+u[7],0))}, total employment ${fmtE(list.reduce((s,u)=>s+u[8],0))}`;
      if (riskCount) { insight = `${riskCount} ${intent==="supplier_search"?"suppliers":"buyers"} are RED PCB outside estates - compliance risk`; }
      else { insight = `All ${intent==="supplier_search"?"suppliers":"buyers"} have acceptable risk profiles`; }
    }
    else if ((intent === "supplier_search" || intent === "buyer_search") && !targetCompany) {
      // No target company specified - run supply_demand analysis instead
      const analysis = {};
      const srcPool = filters.sector ? filterBySector(units, filters.sector) : units;
      srcPool.forEach(u => {
        const d = districts[u[2]] || "Unknown";
        if (!analysis[d]) analysis[d] = { demand:0, supply:0, demandInv:0, supplyInv:0, dUnits:[], sUnits:[] };
        if (u[4] >= 3) { analysis[d].demand++; analysis[d].demandInv+=u[7]; analysis[d].dUnits.push(u); }
        else { analysis[d].supply++; analysis[d].supplyInv+=u[7]; analysis[d].sUnits.push(u); }
      });
      const gaps = Object.entries(analysis).filter(([,v])=>v.demand+v.supply>0)
        .map(([d,v])=>({d, ...v, ratio: v.supply>0?(v.demand/v.supply).toFixed(2):"no supply"}))
        .sort((a,b)=>(parseFloat(b.ratio)||999)-(parseFloat(a.ratio)||999));
      // Show top buyer-supplier pairs per district
      const pairLinks = [];
      gaps.slice(0,8).forEach(g => {
        const topB = g.dUnits.sort((a,b)=>b[7]-a[7])[0];
        const topS = g.sUnits.sort((a,b)=>b[7]-a[7])[0];
        if (topB && topS) pairLinks.push({ from: topS[1], to: topB[1], type: "supplier" });
      });
      const allAnchors = gaps.flatMap(g => [...g.dUnits.slice(0,2), ...g.sUnits.slice(0,1)]).sort((a,b)=>b[7]-a[7]);
      companies = allAnchors.slice(0, maxResults).map(toCompany);
      links = pairLinks;
      prose = `Supply-demand linkage analysis across ${gaps.length} districts:\n\n` +
        gaps.slice(0,10).map(g => `  ${g.d}: ${g.demand} buyers + ${g.supply} suppliers (ratio ${g.ratio})`).join("\n") +
        `\n\nTop buyer-supplier pairs identified in the links below.`;
      insight = `${pairLinks.length} potential buyer-supplier connections identified across districts`;
      recommendation = "Click any company to view its full network graph with suppliers and buyers";
      groupData = { type:"district_bar", data: gaps.slice(0,12).map(g=>({label:g.d, value:g.demand, sub:`S:${g.supply}`})) };
    }
    else if (intent === "risk_assessment") {
      let riskPool = pool.length ? pool : units;
      if (!filters.pcb) riskPool = riskPool.filter(u => u[5] === 2);
      if (!filters.outsideEstate) riskPool = riskPool.filter(u => u[10] === 0);
      riskPool.sort((a,b) => b[7]-a[7]);
      companies = riskPool.slice(0, maxResults).map(toCompany);
      const totalInv = riskPool.reduce((s,u)=>s+u[7],0);
      const totalEmp = riskPool.reduce((s,u)=>s+u[8],0);
      const byDist = {};
      riskPool.forEach(u => { const d = districts[u[2]]; byDist[d] = (byDist[d]||0)+1; });
      const topDist = Object.entries(byDist).sort((a,b)=>b[1]-a[1]).slice(0,5);
      prose = `Found ${riskPool.length} high-risk companies (RED PCB, outside industrial estates).\n\nTotal investment at risk: ${fmt(totalInv)}\nTotal employment affected: ${fmtE(totalEmp)}\n\nTop risk districts:\n${topDist.map(([d,c]) => `  ${d}: ${c} companies`).join("\n")}`;
      insight = `${riskPool.length} companies with ${fmt(totalInv)} investment face environmental compliance risk`;
      recommendation = `Priority: relocate or upgrade ${topDist[0]?topDist[0][0]:""} companies first (${topDist[0]?topDist[0][1]:0} units)`;
      groupData = { type: "district_bar", data: topDist.map(([d,c]) => ({ label: d, value: c })) };
    }
    else if (intent === "cluster_analysis") {
      const mandalCounts = {};
      const targetPool = pool.length ? pool : units;
      targetPool.forEach(u => {
        const key = u[3];
        if (!mandalCounts[key]) mandalCounts[key] = { count:0, inv:0, emp:0, units:[] };
        mandalCounts[key].count++;
        mandalCounts[key].inv += u[7];
        mandalCounts[key].emp += u[8];
        mandalCounts[key].units.push(u);
      });
      const clusters = Object.entries(mandalCounts)
        .filter(([,v]) => v.count >= 5)
        .sort((a,b) => b[1].count - a[1].count);
      const clusterUnits = clusters.flatMap(([,v]) => v.units).sort((a,b) => b[7]-a[7]);
      companies = clusterUnits.slice(0, maxResults).map(toCompany);
      prose = `Found ${clusters.length} production clusters (5+ units in same mandal).\n\n${clusters.slice(0,10).map(([mIdx,v]) => `  ${mandals[parseInt(mIdx)]||"Unknown"} (${districts[v.units[0][2]]}): ${v.count} units, total inv ${fmt(v.inv)}, emp ${fmtE(v.emp)}`).join("\n")}`;
      insight = `${clusters.length} mandals form natural production corridors with ${clusterUnits.length} total units`;
      groupData = { type: "cluster_bar", data: clusters.slice(0,10).map(([mIdx,v]) => ({ label: mandals[parseInt(mIdx)]||"?", value: v.count, sub: districts[v.units[0][2]] })) };
    }
    else if (intent === "diversity") {
      // pool already has sector/district/category filters applied; just ensure women filter
      if (!filters.women) pool = pool.filter(u => u[11] === 1);
      const sectorLabel = filters.sector ? SECTORS[filters.sector].label : null;
      const catLabel = filters.categories ? filters.categories.join("/") : (filters.category || null);
      companies = pool.slice(0, maxResults).map(toCompany);
      const byDist = {};
      pool.forEach(u => { const d = districts[u[2]]; byDist[d] = (byDist[d]||0)+1; });
      const topDist = Object.entries(byDist).sort((a,b)=>b[1]-a[1]).slice(0,8);
      prose = `Found ${pool.length} women-led${sectorLabel ? " "+sectorLabel : ""} companies${catLabel ? " ("+catLabel+")" : ""}${filters.district ? " in "+filters.district : ""}.\n\nTotal investment: ${fmt(pool.reduce((s,u)=>s+u[7],0))}\nTotal employment: ${fmtE(pool.reduce((s,u)=>s+u[8],0))}\n\nTop ${Math.min(5,pool.length)} by investment:\n${pool.slice(0,5).map((u,i) => `  ${i+1}. ${u[1]} - ${districts[u[2]]}, inv ${fmt(u[7])}, emp ${fmtE(u[8])}`).join("\n")}\n\nDistrict spread:\n${topDist.map(([d,c]) => `  ${d}: ${c}`).join("\n")}`;
      insight = `${pool.length} women-led companies with total ${fmt(pool.reduce((s,u)=>s+u[7],0))} investment`;
      groupData = { type: "district_bar", data: topDist.map(([d,c]) => ({ label: d, value: c })) };
    }
    else if (intent === "odop_analysis") {
      if (!filters.odop) pool = pool.filter(u => u[12] === 1);
      companies = pool.slice(0, maxResults).map(toCompany);
      const byDist = {};
      pool.forEach(u => { const d = districts[u[2]]; byDist[d] = (byDist[d]||0)+1; });
      const ranked = Object.entries(byDist).sort((a,b)=>b[1]-a[1]);
      prose = `Found ${pool.length} ODOP companies${turnMatch ? " with turnover above "+turnMatch[1]+" lakhs" : ""}.\n\nRanked by district:\n${ranked.slice(0,10).map(([d,c],i) => `  ${i+1}. ${d}: ${c} companies`).join("\n")}`;
      insight = `${ranked[0]?ranked[0][0]:""} leads with ${ranked[0]?ranked[0][1]:0} ODOP units`;
      groupData = { type: "district_bar", data: ranked.slice(0,10).map(([d,c]) => ({ label: d, value: c })) };
    }
    else if (intent === "acquisition") {
      let acqPool = pool.filter(u => u[4] <= 1 && u[7] >= 100);
      if (filters.minInvestment) acqPool = acqPool.filter(u => u[7] >= filters.minInvestment);
      acqPool.sort((a,b) => b[7]-a[7]);
      companies = acqPool.slice(0, maxResults).map(toCompany);
      prose = `Found ${acqPool.length} potential acquisition targets (MICRO/SMALL with investment >= ${fmt(filters.minInvestment||100)}).\n\nThese are high-investment small companies that could scale with acquisition capital.\nTop ${Math.min(5,acqPool.length)} by investment:\n${acqPool.slice(0,5).map((u,i) => `  ${i+1}. ${u[1]} - ${districts[u[2]]}, inv ${fmt(u[7])}, emp ${fmtE(u[8])}`).join("\n")}`;
      insight = `${acqPool.length} small companies have outsized investment - potential growth plays`;
    }
    else if (intent === "ranking") {
      // District-level ranking
      const byDist = {};
      const rankPool = pool.length < units.length ? pool : (filters.category ? pool : units);
      rankPool.forEach(u => {
        const d = districts[u[2]];
        if (!byDist[d]) byDist[d] = { count:0, inv:0, emp:0 };
        byDist[d].count++;
        byDist[d].inv += u[7];
        byDist[d].emp += u[8];
      });
      const ranked = Object.entries(byDist).sort((a,b) => b[1].count - a[1].count);
      companies = rankPool.slice(0, maxResults).map(toCompany);
      prose = `District ranking${filters.category ? " for "+filters.category+" companies" : ""}:\n\n${ranked.slice(0,15).map(([d,v],i) => `  ${i+1}. ${d}: ${v.count} companies, inv ${fmt(v.inv)}, emp ${fmtE(v.emp)}`).join("\n")}`;
      insight = `${ranked[0]?ranked[0][0]:""} has the highest concentration with ${ranked[0]?ranked[0][1].count:0} units`;
      groupData = { type: "district_bar", data: ranked.slice(0,10).map(([d,v]) => ({ label: d, value: v.count })) };
    }
    else {
      // General / search - sector-aware results with stats
      const sectorLabel = filters.sector ? SECTORS[filters.sector].label : null;
      companies = pool.slice(0, maxResults).map(toCompany);
      const totalInv = pool.reduce((s,u)=>s+u[7],0);
      const totalEmp = pool.reduce((s,u)=>s+u[8],0);
      const totalTurn = pool.reduce((s,u)=>s+u[9],0);
      const catBreakdown = {};
      pool.forEach(u => { const c = CATS[u[4]]; catBreakdown[c] = (catBreakdown[c]||0)+1; });
      const pcbBreakdown = {};
      pool.forEach(u => { const p = PCBS[u[5]]; pcbBreakdown[p] = (pcbBreakdown[p]||0)+1; });
      const byDist = {};
      pool.forEach(u => { const d = districts[u[2]]; byDist[d] = (byDist[d]||0)+1; });
      const topDist = Object.entries(byDist).sort((a,b)=>b[1]-a[1]).slice(0,10);
      const redRisk = pool.filter(u=>u[5]===2&&u[10]===0).length;
      const womenCount = pool.filter(u=>u[11]===1).length;
      const estateCount = pool.filter(u=>u[10]===1).length;

      prose = `Found ${pool.length.toLocaleString()} ${sectorLabel ? sectorLabel : ""} companies${filters.district ? " in "+filters.district : ""}${filters.category ? " ("+filters.category+")" : ""}.\n\nTotal investment: ${fmt(totalInv)}\nTotal employment: ${fmtE(totalEmp)}\nTotal turnover: ${fmt(totalTurn)}\n\nBy size: ${Object.entries(catBreakdown).map(([c,n])=>`${c}: ${n}`).join(", ")}\nBy PCB: ${Object.entries(pcbBreakdown).map(([p,n])=>`${p}: ${n}`).join(", ")}\nWomen-led: ${womenCount} (${(womenCount/pool.length*100).toFixed(1)}%)\nIn industrial estate: ${estateCount} (${(estateCount/pool.length*100).toFixed(1)}%)${filters.sector ? "" : "\nODOP: "+pool.filter(u=>u[12]===1).length}\n\nTop districts:\n${topDist.map(([d,c]) => `  ${d}: ${c}`).join("\n")}`;
      insight = pool.length
        ? `Largest: ${pool[0][1]} (${fmt(pool[0][7])} investment, ${fmtE(pool[0][8])} workers)`
        : "No companies match this query - try different keywords";
      if (redRisk > 0) recommendation = `${redRisk} companies are RED PCB outside estates - environmental compliance risk`;
      else if (womenCount < pool.length * 0.1) recommendation = `Only ${(womenCount/pool.length*100).toFixed(1)}% women-led - diversity opportunity`;
      groupData = { type: "district_bar", data: topDist.map(([d,c]) => ({ label: d, value: c })) };
    }

    const contextInfo = { intent, filters, totalMatching: pool.length };
    if (p.limit) contextInfo.limit = p.limit;
    if (qSortBy !== "inv") contextInfo.sortBy = qSortBy;
    if (exclude.district) contextInfo.excludeDistrict = exclude.district;
    if (exclude.category) contextInfo.excludeCategory = exclude.category;

    setAiResult({
      prose, query: qText,
      data: { companies, links, insight, recommendation },
      context: contextInfo,
      groupData,
    });
  }, [units, districts, mandals]);

  // runQuery: calls ML API first, falls back to regex parser
  const runQuery = useCallback(async (q) => {
    const qText = q || aiQuery;
    if (!qText.trim()) return;
    setTab("ai");

    try {
      const res = await fetch("/api/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: qText }),
      });
      if (res.ok) {
        const parsed = await res.json();
        parsed._source = "ml";
        executeQuery(qText, parsed);
        return;
      }
    } catch(e) {}

    // Fallback: regex parser
    const parsed = parseQuery(qText, units, districts, mandals);
    const p = {
      intent: parsed.intent,
      sector: parsed.filters?.sector,
      district: parsed.filters?.district,
      category: parsed.filters?.category,
      categories: parsed.filters?.categories,
      pcb: parsed.filters?.pcb,
      women: parsed.filters?.women,
      odop: parsed.filters?.odop,
      outside_estate: parsed.filters?.outsideEstate,
      in_estate: parsed.filters?.inEstate,
      min_investment: parsed.filters?.minInvestment,
      min_turnover: parsed.filters?.minTurnover,
      min_employment: parsed.filters?.minEmployment,
      limit: parsed.limit,
      sort_by: parsed.sortBy,
      target_company: parsed.targetCompany ? parsed.targetCompany[1] : null,
      exclude_district: parsed.exclude?.district,
      exclude_category: parsed.exclude?.category,
      _source: "regex",
    };
    executeQuery(qText, p);
  }, [aiQuery, units, districts, mandals, executeQuery]);

  // ????????? STYLES ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
  const C = {
    wrap: { fontFamily:"system-ui,sans-serif", background:"#f5f5f5", minHeight:"100vh", color:"#1a1a1a" },
    hdr: { background:"#fff", borderBottom:"1px solid #e5e5e5", padding:"0 20px", display:"flex", alignItems:"center", gap:20 },
    logo: { fontSize:14, fontWeight:600, padding:"14px 0", whiteSpace:"nowrap" },
    tabs: { display:"flex", gap:0, flex:1 },
    tabBtn: (a) => ({ padding:"14px 16px", fontSize:13, fontWeight:a?500:400, color:a?"#1a1a1a":"#888", borderBottom:a?"2px solid #0C4E64":"2px solid transparent", cursor:"pointer", background:"none", border:"none", borderRadius:0 }),
    badge: (col) => ({ display:"inline-block", padding:"2px 8px", borderRadius:10, fontSize:11, fontWeight:500, background:col+"22", color:col, marginRight:4, whiteSpace:"nowrap" }),
    card: { background:"#fff", borderRadius:10, padding:16, border:"1px solid #e5e5e5", marginBottom:14 },
    filterRow: { display:"flex", gap:8, flexWrap:"wrap", alignItems:"center", padding:"10px 20px", background:"#fafafa", borderBottom:"1px solid #e5e5e5" },
    sel: { padding:"5px 8px", fontSize:12, borderRadius:6, border:"1px solid #ddd", background:"#fff", color:"#1a1a1a", cursor:"pointer" },
    togBtn: (a) => ({ padding:"5px 12px", fontSize:12, borderRadius:16, background:a?"#0C4E64":"transparent", color:a?"#fff":"#888", border:a?"none":"1px solid #ddd", cursor:"pointer", fontWeight:a?500:400 }),
    main: { padding:"16px 20px", maxWidth:1200, margin:"0 auto" },
    grid2: { display:"grid", gridTemplateColumns:"1fr 360px", gap:14 },
    table: { width:"100%", borderCollapse:"collapse", fontSize:12 },
    th: { textAlign:"left", padding:"8px 10px", borderBottom:"1px solid #e5e5e5", color:"#888", fontSize:11, fontWeight:500, textTransform:"uppercase", whiteSpace:"nowrap" },
    td: { padding:"8px 10px", borderBottom:"1px solid #e5e5e5", verticalAlign:"middle" },
    compName: { fontWeight:500, fontSize:13, cursor:"pointer", color:"#1a1a1a" },
    compSub: { fontSize:11, color:"#888", marginTop:1 },
    aiInput: { flex:1, padding:"11px 14px", fontSize:13, borderRadius:8, border:"1.5px solid #ddd", background:"#fff", color:"#1a1a1a", outline:"none" },
    aiBtn: { padding:"11px 18px", background:"#0C4E64", color:"#fff", border:"none", borderRadius:8, fontSize:13, fontWeight:500, cursor:"pointer", flexShrink:0 },
    chip: () => ({ padding:"5px 10px", fontSize:11, borderRadius:16, border:"1px solid #ddd", background:"transparent", color:"#888", cursor:"pointer", textAlign:"left" }),
    linkCard: (type) => ({ padding:"10px 12px", borderRadius:8, border:"1px solid #e5e5e5", marginBottom:8, borderLeft:`3px solid ${type==="supplier"?"#1D9E75":type==="buyer"?"#378ADD":"#E24B4A"}` }),
    kpi: { background:"#fafafa", borderRadius:8, padding:"12px 14px", border:"1px solid #e5e5e5", textAlign:"center" },
    kpiV: { fontSize:20, fontWeight:600, lineHeight:1 },
    kpiL: { fontSize:11, color:"#888", marginTop:3 },
  };

  const DIST_LIST = ["ALL", ...districts.filter(d=>d!=="Unknown")];

  return (
    <div style={C.wrap}>
      {/* Header */}
      <div style={C.hdr}>
        <div style={C.logo}>MSME <span style={{color:"#0C4E64"}}>Supply Chain</span> Intelligence</div>
        <div style={C.tabs}>
          {[["search","Company search"],["ai","AI analyst"],["map","Supply map"],["network","Network graph"],["links","Supply links"]].map(([k,l]) => (
            <button key={k} style={C.tabBtn(tab===k)} onClick={()=>setTab(k)}>{l}</button>
          ))}
        </div>
        <div style={{fontSize:10, padding:"4px 10px", borderRadius:12, background:"#1D9E7522", color:"#1D9E75", fontWeight:500}}>
          ML-powered
        </div>
        <div style={{fontSize:11, color:"#888", whiteSpace:"nowrap"}}>
          {filtered.length.toLocaleString()} / {units.length.toLocaleString()} companies
        </div>
      </div>

      {/* Filter bar */}
      <div style={C.filterRow}>
        <input value={query} onChange={e=>{setQuery(e.target.value);setPage(0)}}
          placeholder="Search company name..."
          style={{...C.aiInput, padding:"5px 10px", fontSize:12, flex:"0 0 200px"}} />
        <select style={C.sel} value={filterDist} onChange={e=>{setFilterDist(e.target.value);setFilterMandal("ALL");setPage(0)}}>
          {DIST_LIST.map(d=><option key={d}>{d}</option>)}
        </select>
        {filterDist !== "ALL" && (
          <select style={C.sel} value={filterMandal} onChange={e=>{setFilterMandal(e.target.value);setPage(0)}}>
            <option value="ALL">All mandals</option>
            {[...new Set(units.filter(u=>districts[u[2]]===filterDist).map(u=>mandals[u[3]]))].filter(Boolean).sort().map(m=>
              <option key={m}>{m}</option>
            )}
          </select>
        )}
        <select style={C.sel} value={filterCat} onChange={e=>{setFilterCat(e.target.value);setPage(0)}}>
          <option value="ALL">All sizes</option>
          {CATS.map(c=><option key={c}>{c}</option>)}
        </select>
        <select style={C.sel} value={filterPcb} onChange={e=>{setFilterPcb(e.target.value);setPage(0)}}>
          <option value="ALL">All PCB</option>
          {PCBS.map(p=><option key={p}>{p}</option>)}
        </select>
        <button style={C.togBtn(filterWomen)} onClick={()=>{setFilterWomen(!filterWomen);setPage(0)}}>Women-led</button>
        <button style={C.togBtn(filterOdop)} onClick={()=>{setFilterOdop(!filterOdop);setPage(0)}}>ODOP</button>
        <button style={C.togBtn(filterEstate)} onClick={()=>{setFilterEstate(!filterEstate);setPage(0)}}>In estate</button>
        <select style={{...C.sel}} value={sortBy} onChange={e=>setSortBy(e.target.value)}>
          <option value="inv">Sort: Investment</option>
          <option value="emp">Sort: Employment</option>
          <option value="turn">Sort: Turnover</option>
          <option value="name">Sort: Name</option>
        </select>
      </div>
      {/* Advanced filters row */}
      <div style={{...C.filterRow, borderTop:"none", paddingTop:0}}>
        <span style={{fontSize:11,color:"#888"}}>Investment (lakhs):</span>
        <input value={filterMinInv} onChange={e=>{setFilterMinInv(e.target.value);setPage(0)}}
          placeholder="Min" type="number" style={{...C.sel, width:70}} />
        <span style={{fontSize:11,color:"#ccc"}}>to</span>
        <input value={filterMaxInv} onChange={e=>{setFilterMaxInv(e.target.value);setPage(0)}}
          placeholder="Max" type="number" style={{...C.sel, width:70}} />
        <span style={{fontSize:11,color:"#888",marginLeft:8}}>Min employment:</span>
        <input value={filterMinEmp} onChange={e=>{setFilterMinEmp(e.target.value);setPage(0)}}
          placeholder="e.g. 50" type="number" style={{...C.sel, width:70}} />
        <div style={{marginLeft:"auto",display:"flex",gap:6,alignItems:"center"}}>
          <span style={{fontSize:12,fontWeight:500,color:"#0C4E64"}}>{filtered.length.toLocaleString()}</span>
          <span style={{fontSize:11,color:"#888"}}>/ {units.length.toLocaleString()}</span>
          {(query || filterDist!=="ALL" || filterCat!=="ALL" || filterPcb!=="ALL" || filterWomen || filterOdop || filterEstate || filterMinInv || filterMaxInv || filterMinEmp || filterMandal!=="ALL") && (
            <button style={{...C.sel,fontSize:10,padding:"3px 8px",color:"#E24B4A",borderColor:"#E24B4A"}}
              onClick={()=>{setQuery("");setFilterDist("ALL");setFilterCat("ALL");setFilterPcb("ALL");setFilterWomen(false);setFilterOdop(false);setFilterEstate(false);setFilterMinInv("");setFilterMaxInv("");setFilterMinEmp("");setFilterMandal("ALL");setPage(0);}}>
              Clear all
            </button>
          )}
        </div>
      </div>

      <div style={C.main}>

        {/* ?????? SEARCH TAB ?????? */}
        {tab === "search" && (
          <div style={selected ? C.grid2 : {}}>
            {/* Company table */}
            <div>
              <div style={C.card}>
                <table style={C.table}>
                  <thead>
                    <tr>
                      <th style={C.th}>Company</th>
                      <th style={C.th}>District / Mandal</th>
                      <th style={C.th}>Size</th>
                      <th style={C.th}>PCB</th>
                      <th style={C.th}>Investment</th>
                      <th style={C.th}>Employment</th>
                      <th style={C.th}>Turnover</th>
                      <th style={C.th}>Flags</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginated.map(u => {
                      const isActive = selected?.[0]===u[0];
                      return (
                        <tr key={u[0]} onClick={()=>setSelected(isActive?null:u)}
                          style={{cursor:"pointer", background:isActive?"#0C4E6410":undefined}}>
                          <td style={C.td}>
                            <div style={C.compName}>{u[1]}</div>
                          </td>
                          <td style={C.td}>
                            <div style={{fontSize:12}}>{districts[u[2]]||"--"}</div>
                            <div style={C.compSub}>{mandals[u[3]]||"--"}</div>
                          </td>
                          <td style={C.td}>
                            <span style={C.badge(CAT_COL[u[4]])}>{CATS[u[4]]}</span>
                          </td>
                          <td style={C.td}>
                            <span style={C.badge(PCB_COL[u[5]])}>{PCBS[u[5]]}</span>
                          </td>
                          <td style={{...C.td, fontWeight:500, fontSize:12}}>{fmt(u[7])}</td>
                          <td style={{...C.td, fontSize:12}}>{fmtE(u[8])}</td>
                          <td style={{...C.td, fontSize:12}}>{fmt(u[9])}</td>
                          <td style={C.td}>
                            {u[11]===1 && <span title="Women-led" style={{...C.badge("#D85A30"), padding:"2px 5px"}}>W</span>}
                            {u[12]===1 && <span title="ODOP" style={{...C.badge("#534AB7"), padding:"2px 5px"}}>O</span>}
                            {u[10]===1 && <span title="In industrial estate" style={{...C.badge("#1D9E75"), padding:"2px 5px"}}>E</span>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>

                {/* Pagination */}
                <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", marginTop:12, fontSize:12, color:"#888"}}>
                  <span>{filtered.length.toLocaleString()} companies found</span>
                  <div style={{display:"flex", gap:6, alignItems:"center"}}>
                    <button disabled={page===0} onClick={()=>setPage(p=>p-1)} style={{...C.sel, cursor:page===0?"not-allowed":"pointer", opacity:page===0?0.4:1}}>Prev</button>
                    <span>Page {page+1} of {Math.max(1,totalPages)}</span>
                    <button disabled={page>=totalPages-1} onClick={()=>setPage(p=>p+1)} style={{...C.sel, cursor:page>=totalPages-1?"not-allowed":"pointer", opacity:page>=totalPages-1?0.4:1}}>Next</button>
                  </div>
                </div>
              </div>
            </div>

            {/* Company detail panel */}
            {selected && (() => {
              const u = selected;
              const { similar, mandal_peers, suppliers } = getCompanyDetail(u);
              return (
                <div>
                  <div style={C.card}>
                    <div style={{display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:12}}>
                      <div style={{fontSize:15, fontWeight:600, lineHeight:1.3, flex:1}}>{u[1]}</div>
                      <button onClick={()=>setSelected(null)} style={{background:"none",border:"none",fontSize:18,cursor:"pointer",color:"#888",padding:"0 0 0 8px"}}>??</button>
                    </div>

                    <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginBottom:14}}>
                      {[
                        ["District", districts[u[2]]||"--"],
                        ["Mandal", mandals[u[3]]||"--"],
                        ["Category", CATS[u[4]]],
                        ["PCB risk", PCBS[u[5]]+" - "+PCB_LABEL[u[5]]],
                        ["Investment", fmt(u[7])],
                        ["Employment", fmtE(u[8])],
                        ["Turnover", fmt(u[9])],
                        ["Social", SOCS[u[6]]||"--"],
                        ["In estate", u[10]?"Yes":"No"],
                        ["Women-led", u[11]?"Yes":"No"],
                        ["ODOP unit", u[12]?"Yes":"No"],
                        ["Lat/Lon", u[13]&&u[13]!==0 ? `${u[13]}, ${u[14]}` : "--"],
                      ].map(([l,v])=>(
                        <div key={l} style={C.kpi}>
                          <div style={{fontSize:10, color:"#888", marginBottom:2}}>{l}</div>
                          <div style={{fontSize:13, fontWeight:500}}>{v}</div>
                        </div>
                      ))}
                    </div>

                    <div style={{display:"flex",gap:8,marginBottom:12}}>
                      <button onClick={()=>{const qry=`Compare ${u[1]} and similar companies`;setAiQuery(qry);runQuery(qry);}}
                        style={{...C.aiBtn, flex:1}}>
                        Analyse
                      </button>
                      <button onClick={()=>openNetwork(u)}
                        style={{...C.aiBtn, flex:1, background:"#1D9E75"}}>
                        Network graph
                      </button>
                    </div>
                  </div>

                  {/* Potential suppliers */}
                  {suppliers.length > 0 && (
                    <div style={C.card}>
                      <div style={{fontSize:12, fontWeight:500, color:"#888", marginBottom:10, textTransform:"uppercase", letterSpacing:"0.5px"}}>Potential suppliers in {districts[u[2]]||"district"}</div>
                      {suppliers.map(s=>(
                        <div key={s[0]} style={{...C.linkCard("supplier"), cursor:"pointer"}} onClick={()=>setSelected(s)}>
                          <div style={{fontSize:12, fontWeight:500}}>{s[1]}</div>
                          <div style={{fontSize:11, color:"#888", marginTop:2}}>
                            {CATS[s[4]]} . {mandals[s[3]]||"--"} . {fmt(s[7])} inv . {fmtE(s[8])} workers
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Mandal peers */}
                  {mandal_peers.length > 0 && (
                    <div style={C.card}>
                      <div style={{fontSize:12, fontWeight:500, color:"#888", marginBottom:10, textTransform:"uppercase", letterSpacing:"0.5px"}}>Cluster peers in {mandals[u[3]]||"mandal"}</div>
                      {mandal_peers.map(s=>(
                        <div key={s[0]} style={{...C.linkCard("buyer"), cursor:"pointer"}} onClick={()=>setSelected(s)}>
                          <div style={{fontSize:12, fontWeight:500}}>{s[1]}</div>
                          <div style={{fontSize:11, color:"#888", marginTop:2}}>
                            {CATS[s[4]]} . <span style={{color:PCB_COL[s[5]]}}>{PCBS[s[5]]}</span> . {fmt(s[7])} inv
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Similar companies */}
                  {similar.length > 0 && (
                    <div style={C.card}>
                      <div style={{fontSize:12, fontWeight:500, color:"#888", marginBottom:10, textTransform:"uppercase", letterSpacing:"0.5px"}}>Similar {CATS[u[4]]} companies in district</div>
                      {similar.map(s=>(
                        <div key={s[0]} style={{...C.linkCard("buyer"), cursor:"pointer"}} onClick={()=>setSelected(s)}>
                          <div style={{fontSize:12, fontWeight:500}}>{s[1]}</div>
                          <div style={{fontSize:11, color:"#888", marginTop:2}}>
                            {mandals[s[3]]||"--"} . {fmt(s[7])} inv . {fmtE(s[8])} workers
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        )}

                {/* ?????? ANALYTICS TAB ??? pattern-based */}
        {tab === "ai" && (() => {
          // Derive analytical properties for each company
          const classify = (u) => {
            const invPerEmp = u[8] > 0 ? u[7] / u[8] : u[7];
            const density = mandalDensity[u[3]] || 0;
            return {
              risk: u[5]===2 && u[10]===0 ? "high" : u[5]===2 ? "moderate" : u[5]===1 && u[10]===0 ? "moderate" : "low",
              intensity: invPerEmp > 50 ? "capital" : invPerEmp > 10 ? "balanced" : "labor",
              scale: (u[4]<=1 && u[7]>500) ? "high_potential" : u[4]>=3 ? "anchor" : u[4]===2 ? "established" : "base",
              density: density >= 100 ? "mega_cluster" : density >= 30 ? "cluster" : density >= 10 ? "group" : "isolated",
              employer: u[8]>=500 ? "major" : u[8]>=100 ? "significant" : u[8]>=20 ? "moderate" : "micro",
            };
          };

          // Apply analytical filters
          let data = units.filter(u => {
            if (aDist !== "ALL" && districts[u[2]] !== aDist) return false;
            const c = classify(u);
            if (aRisk !== "all" && c.risk !== aRisk) return false;
            if (aIntensity !== "all" && c.intensity !== aIntensity) return false;
            if (aScale !== "all" && c.scale !== aScale) return false;
            if (aDensity !== "all" && c.density !== aDensity) return false;
            if (aEmployer !== "all" && c.employer !== aEmployer) return false;
            return true;
          });
          data.sort((a,b) => b[7] - a[7]);

          const totalInv = data.reduce((s,u)=>s+u[7],0);
          const totalEmp = data.reduce((s,u)=>s+u[8],0);
          const totalTurn = data.reduce((s,u)=>s+u[9],0);

          // Breakdowns
          const byDist = {}; data.forEach(u => { const d = districts[u[2]]||"?"; byDist[d] = (byDist[d]||0)+1; });
          const distRanked = Object.entries(byDist).sort((a,b)=>b[1]-a[1]);
          const byCat = {}; data.forEach(u => { byCat[CATS[u[4]]] = (byCat[CATS[u[4]]]||0)+1; });
          const byPcb = {}; data.forEach(u => { byPcb[PCBS[u[5]]] = (byPcb[PCBS[u[5]]]||0)+1; });
          const womenCount = data.filter(u=>u[11]===1).length;
          const estateCount = data.filter(u=>u[10]===1).length;
          const redOutside = data.filter(u=>u[5]===2&&u[10]===0).length;

          // Classification distribution
          const riskDist = {}; const intDist = {}; const scaleDist = {}; const densDist = {}; const empDist = {};
          data.forEach(u => {
            const c = classify(u);
            riskDist[c.risk] = (riskDist[c.risk]||0)+1;
            intDist[c.intensity] = (intDist[c.intensity]||0)+1;
            scaleDist[c.scale] = (scaleDist[c.scale]||0)+1;
            densDist[c.density] = (densDist[c.density]||0)+1;
            empDist[c.employer] = (empDist[c.employer]||0)+1;
          });

          const hasAnyFilter = aRisk!=="all"||aIntensity!=="all"||aScale!=="all"||aDensity!=="all"||aEmployer!=="all"||aDist!=="ALL";

          const FilterBtn = ({value, current, setter, color, label, count}) => (
            <button onClick={()=>setter(current===value?"all":value)}
              style={{padding:"4px 10px",fontSize:11,borderRadius:12,cursor:"pointer",
                background:current===value?(color||"#0C4E64"):undefined,
                color:current===value?"#fff":"#555",
                border:current===value?"none":"1px solid #ddd",
                fontWeight:current===value?500:400}}>
              {label} {count !== undefined && <span style={{opacity:0.7}}>({count.toLocaleString()})</span>}
            </button>
          );

          return (
            <>
              {/* Analytical filters */}
              <div style={C.card}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
                  <div style={{fontSize:13,fontWeight:500}}>Pattern-based analysis</div>
                  {hasAnyFilter && (
                    <button style={{...C.sel,fontSize:10,color:"#E24B4A",borderColor:"#E24B4A"}}
                      onClick={()=>{setARisk("all");setAIntensity("all");setAScale("all");setADensity("all");setAEmployer("all");setADist("ALL");}}>
                      Clear all
                    </button>
                  )}
                </div>

                <div style={{marginBottom:8}}>
                  <div style={{fontSize:11,color:"#888",marginBottom:4}}>Risk profile</div>
                  <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
                    <FilterBtn value="low" current={aRisk} setter={setARisk} color="#1D9E75" label="Low risk" count={riskDist.low}/>
                    <FilterBtn value="moderate" current={aRisk} setter={setARisk} color="#EF9F27" label="Moderate" count={riskDist.moderate}/>
                    <FilterBtn value="high" current={aRisk} setter={setARisk} color="#E24B4A" label="High risk" count={riskDist.high}/>
                  </div>
                  <div style={{fontSize:10,color:"#aaa",marginTop:2}}>Based on PCB tier + industrial estate status</div>
                </div>

                <div style={{marginBottom:8}}>
                  <div style={{fontSize:11,color:"#888",marginBottom:4}}>Investment intensity</div>
                  <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
                    <FilterBtn value="capital" current={aIntensity} setter={setAIntensity} color="#378ADD" label="Capital-heavy" count={intDist.capital}/>
                    <FilterBtn value="balanced" current={aIntensity} setter={setAIntensity} color="#7F77DD" label="Balanced" count={intDist.balanced}/>
                    <FilterBtn value="labor" current={aIntensity} setter={setAIntensity} color="#D85A30" label="Labor-intensive" count={intDist.labor}/>
                  </div>
                  <div style={{fontSize:10,color:"#aaa",marginTop:2}}>Investment per employee: capital (&gt;50L), balanced (10-50L), labor (&lt;10L)</div>
                </div>

                <div style={{marginBottom:8}}>
                  <div style={{fontSize:11,color:"#888",marginBottom:4}}>Scale potential</div>
                  <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
                    <FilterBtn value="anchor" current={aScale} setter={setAScale} color="#E24B4A" label="Anchor (LARGE/MEGA)" count={scaleDist.anchor}/>
                    <FilterBtn value="established" current={aScale} setter={setAScale} color="#EF9F27" label="Established (MEDIUM)" count={scaleDist.established}/>
                    <FilterBtn value="high_potential" current={aScale} setter={setAScale} color="#1D9E75" label="High potential" count={scaleDist.high_potential}/>
                    <FilterBtn value="base" current={aScale} setter={setAScale} color="#888" label="Base tier" count={scaleDist.base}/>
                  </div>
                  <div style={{fontSize:10,color:"#aaa",marginTop:2}}>High potential = MICRO/SMALL with investment &gt;500 lakhs (growth candidates)</div>
                </div>

                <div style={{marginBottom:8}}>
                  <div style={{fontSize:11,color:"#888",marginBottom:4}}>Cluster density</div>
                  <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
                    <FilterBtn value="mega_cluster" current={aDensity} setter={setADensity} color="#0C4E64" label="Mega cluster (100+)" count={densDist.mega_cluster}/>
                    <FilterBtn value="cluster" current={aDensity} setter={setADensity} color="#378ADD" label="Cluster (30-99)" count={densDist.cluster}/>
                    <FilterBtn value="group" current={aDensity} setter={setADensity} color="#7F77DD" label="Group (10-29)" count={densDist.group}/>
                    <FilterBtn value="isolated" current={aDensity} setter={setADensity} color="#888" label="Isolated (<10)" count={densDist.isolated}/>
                  </div>
                  <div style={{fontSize:10,color:"#aaa",marginTop:2}}>Number of companies in the same mandal</div>
                </div>

                <div style={{marginBottom:8}}>
                  <div style={{fontSize:11,color:"#888",marginBottom:4}}>Employer significance</div>
                  <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
                    <FilterBtn value="major" current={aEmployer} setter={setAEmployer} color="#E24B4A" label="Major (500+)" count={empDist.major}/>
                    <FilterBtn value="significant" current={aEmployer} setter={setAEmployer} color="#D85A30" label="Significant (100-499)" count={empDist.significant}/>
                    <FilterBtn value="moderate" current={aEmployer} setter={setAEmployer} color="#EF9F27" label="Moderate (20-99)" count={empDist.moderate}/>
                    <FilterBtn value="micro" current={aEmployer} setter={setAEmployer} color="#888" label="Micro (<20)" count={empDist.micro}/>
                  </div>
                </div>

                <div>
                  <div style={{fontSize:11,color:"#888",marginBottom:4}}>District</div>
                  <select style={C.sel} value={aDist} onChange={e=>setADist(e.target.value)}>
                    <option value="ALL">All districts</option>
                    {[...new Set(units.map(u=>districts[u[2]]))].filter(d=>d&&d!=="Unknown").sort().map(d=><option key={d}>{d}</option>)}
                  </select>
                </div>
              </div>

              {/* KPIs */}
              <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:10,marginBottom:14}}>
                <div style={C.kpi}><div style={C.kpiV}>{data.length.toLocaleString()}</div><div style={C.kpiL}>Companies</div></div>
                <div style={C.kpi}><div style={C.kpiV}>{fmt(totalInv)}</div><div style={C.kpiL}>Investment</div></div>
                <div style={C.kpi}><div style={C.kpiV}>{fmtE(totalEmp)}</div><div style={C.kpiL}>Employment</div></div>
                <div style={C.kpi}><div style={C.kpiV}>{fmt(totalTurn)}</div><div style={C.kpiL}>Turnover</div></div>
                <div style={C.kpi}><div style={C.kpiV}>{data.length?fmt(Math.round(totalInv/data.length)):"--"}</div><div style={C.kpiL}>Avg investment</div></div>
              </div>

              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
                {/* Size breakdown */}
                <div style={C.card}>
                  <div style={{fontSize:11,fontWeight:500,color:"#888",textTransform:"uppercase",marginBottom:10}}>Size distribution</div>
                  {CATS.map((cat,i) => {
                    const n = byCat[cat]||0; const pct = data.length ? n/data.length : 0;
                    return (
                      <div key={cat} style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                        <span style={{fontSize:12,width:60}}>{cat}</span>
                        <div style={{flex:1,height:16,background:"#f0f0f0",borderRadius:3,overflow:"hidden"}}>
                          <div style={{height:"100%",width:`${pct*100}%`,background:CAT_COL[i],borderRadius:3,minWidth:n>0?2:0}}/>
                        </div>
                        <span style={{fontSize:12,fontWeight:500,width:55,textAlign:"right"}}>{n.toLocaleString()}</span>
                        <span style={{fontSize:10,color:"#888",width:35,textAlign:"right"}}>{(pct*100).toFixed(0)}%</span>
                      </div>
                    );
                  })}
                </div>

                {/* PCB breakdown */}
                <div style={C.card}>
                  <div style={{fontSize:11,fontWeight:500,color:"#888",textTransform:"uppercase",marginBottom:10}}>PCB tier</div>
                  {PCBS.map((pcb,i) => {
                    const n = byPcb[pcb]||0; const pct = data.length ? n/data.length : 0;
                    return (
                      <div key={pcb} style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                        <span style={{fontSize:12,width:60}}>{pcb}</span>
                        <div style={{flex:1,height:16,background:"#f0f0f0",borderRadius:3,overflow:"hidden"}}>
                          <div style={{height:"100%",width:`${pct*100}%`,background:PCB_COL[i],borderRadius:3,minWidth:n>0?2:0}}/>
                        </div>
                        <span style={{fontSize:12,fontWeight:500,width:55,textAlign:"right"}}>{n.toLocaleString()}</span>
                        <span style={{fontSize:10,color:"#888",width:35,textAlign:"right"}}>{(pct*100).toFixed(0)}%</span>
                      </div>
                    );
                  })}
                  {redOutside > 0 && (
                    <div style={{marginTop:8,padding:"6px 10px",background:"#E24B4A11",borderRadius:6,fontSize:12,color:"#E24B4A",borderLeft:"3px solid #E24B4A"}}>
                      {redOutside.toLocaleString()} RED PCB outside estates - compliance risk
                    </div>
                  )}
                </div>

                {/* Flags */}
                <div style={C.card}>
                  <div style={{fontSize:11,fontWeight:500,color:"#888",textTransform:"uppercase",marginBottom:10}}>Key indicators</div>
                  {[
                    ["Women-led", womenCount, "#D85A30"],
                    ["In industrial estate", estateCount, "#1D9E75"],
                    ["ODOP", data.filter(u=>u[12]===1).length, "#534AB7"],
                  ].map(([label, n, col]) => {
                    const pct = data.length ? n/data.length : 0;
                    return (
                      <div key={label} style={{display:"flex",alignItems:"center",gap:8,marginBottom:8}}>
                        <span style={{fontSize:12,flex:1}}>{label}</span>
                        <div style={{width:100,height:14,background:"#f0f0f0",borderRadius:3,overflow:"hidden"}}>
                          <div style={{height:"100%",width:`${pct*100}%`,background:col,borderRadius:3,minWidth:n>0?2:0}}/>
                        </div>
                        <span style={{fontSize:12,fontWeight:500,width:55,textAlign:"right"}}>{n.toLocaleString()}</span>
                        <span style={{fontSize:10,color:"#888",width:35,textAlign:"right"}}>{(pct*100).toFixed(0)}%</span>
                      </div>
                    );
                  })}
                </div>

                {/* District chart */}
                <div style={C.card}>
                  <div style={{fontSize:11,fontWeight:500,color:"#888",textTransform:"uppercase",marginBottom:10}}>Top districts</div>
                  <svg viewBox={`0 0 400 ${Math.min(distRanked.length,10) * 26 + 5}`} style={{width:"100%"}}>
                    {distRanked.slice(0,10).map(([d,n],i) => {
                      const maxN = distRanked[0]?.[1]||1;
                      const barW = (n/maxN)*240;
                      return (
                        <g key={d} transform={`translate(0,${i*26})`}>
                          <text x={120} y={16} textAnchor="end" fontSize={11} fill="#1a1a1a">{d.length>14?d.slice(0,12)+"..":d}</text>
                          <rect x={128} y={3} width={barW} height={18} rx={3} fill="#0C4E64" opacity={0.7}/>
                          <text x={133+barW} y={16} fontSize={11} fill="#888">{n.toLocaleString()}</text>
                        </g>
                      );
                    })}
                  </svg>
                </div>
              </div>

              {/* Top companies */}
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
                <div style={C.card}>
                  <div style={{fontSize:11,fontWeight:500,color:"#888",textTransform:"uppercase",marginBottom:8}}>Top by investment</div>
                  {data.slice(0,8).map((u,i) => {
                    const c = classify(u);
                    return (
                      <div key={u[0]} style={{display:"flex",alignItems:"center",gap:6,padding:"5px 0",borderTop:i?"1px solid #f5f5f5":"none",cursor:"pointer"}}
                        onClick={()=>{setSelected(u);setTab("search");}}>
                        <span style={{fontSize:10,color:"#888",width:16}}>{i+1}.</span>
                        <span style={{...C.badge(CAT_COL[u[4]]),fontSize:9}}>{CATS[u[4]]}</span>
                        <span style={{fontSize:12,flex:1}}>{u[1].slice(0,28)}</span>
                        <span style={{fontSize:9,color:c.risk==="high"?"#E24B4A":c.risk==="moderate"?"#EF9F27":"#1D9E75",marginRight:4}}>{c.risk}</span>
                        <span style={{fontSize:12,fontWeight:500}}>{fmt(u[7])}</span>
                      </div>
                    );
                  })}
                </div>
                <div style={C.card}>
                  <div style={{fontSize:11,fontWeight:500,color:"#888",textTransform:"uppercase",marginBottom:8}}>Top by employment</div>
                  {[...data].sort((a,b)=>b[8]-a[8]).slice(0,8).map((u,i) => {
                    const c = classify(u);
                    return (
                      <div key={u[0]} style={{display:"flex",alignItems:"center",gap:6,padding:"5px 0",borderTop:i?"1px solid #f5f5f5":"none",cursor:"pointer"}}
                        onClick={()=>{setSelected(u);setTab("search");}}>
                        <span style={{fontSize:10,color:"#888",width:16}}>{i+1}.</span>
                        <span style={{...C.badge(CAT_COL[u[4]]),fontSize:9}}>{CATS[u[4]]}</span>
                        <span style={{fontSize:12,flex:1}}>{u[1].slice(0,28)}</span>
                        <span style={{fontSize:9,color:c.intensity==="capital"?"#378ADD":c.intensity==="labor"?"#D85A30":"#7F77DD",marginRight:4}}>{c.intensity}</span>
                        <span style={{fontSize:12,fontWeight:500}}>{fmtE(u[8])}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          );
        })()}

                                                {/* ?????? SUPPLY MAP TAB ??? Data-driven Explorer */}
        {tab === "map" && (() => {
          const SW = 740, SH = 580;
          const TG_FULL = { minLat: 15.65, maxLat: 20.05, minLon: 77.05, maxLon: 81.65 };
          const b = mapView;

          const proj = (lat, lon) => ({
            x: (lon - b.minLon) / (b.maxLon - b.minLon) * SW,
            y: (b.maxLat - lat) / (b.maxLat - b.minLat) * SH,
          });
          const polyPath = (pts) =>
            pts.map((p,i) => { const q=proj(p[0],p[1]); return `${i===0?'M':'L'}${q.x.toFixed(1)},${q.y.toFixed(1)}`; }).join(' ') + ' Z';

          const lonSpan = b.maxLon - b.minLon;

          // Strict Telangana bounds (removes stray GPS outside state)
          const TG_LAT=[15.75,19.95], TG_LON=[77.15,81.65];

          // Cluster merge radius shrinks as you zoom in
          // At lonSpan < 0.012 ? SHOW_ALL, skip clustering entirely
          const SHOW_ALL = lonSpan < 0.012;
          const MERGE_R = lonSpan < 0.04  ? 7
                        : lonSpan < 0.08  ? 10
                        : lonSpan < 0.15  ? 13
                        : lonSpan < 0.30  ? 16
                        : lonSpan < 0.60  ? 20
                        : lonSpan < 1.2   ? 28
                        : lonSpan < 2.5   ? 38 : 52;
          const MERGE_R2 = MERGE_R * MERGE_R;

          // Pre-compute outline path for reuse (fill + stroke + clip)
          const outlinePath = TG_OUTLINE.map((p,i)=>{
            const q=proj(p[0],p[1]);
            return `${i===0?'M':'L'}${q.x.toFixed(1)},${q.y.toFixed(1)}`;
          }).join(' ')+'Z';

          const allPts = [];
          geoUnits.forEach(u => {
            const p = proj(u[13],u[14]);
            if (p.x>=-4 && p.x<=SW+4 && p.y>=-4 && p.y<=SH+4) allPts.push({x:p.x,y:p.y,u});
          });

          const clusters = [];
          if (SHOW_ALL) {
            // Every company gets its own dot ? no clustering
            allPts.forEach(pt => clusters.push({
              x:pt.x, y:pt.y, count:1, us:[pt.u],
              minLat:pt.u[13],maxLat:pt.u[13],minLon:pt.u[14],maxLon:pt.u[14],
              domCat:pt.u[4], single:pt.u,
            }));
          } else {
            const assigned = new Uint8Array(allPts.length);
            for (let i=0;i<allPts.length;i++) {
              if (assigned[i]) continue;
              const cl={sx:0,sy:0,n:0,us:[],lats:[],lons:[],ct:[0,0,0,0,0]};
              for (let j=i;j<allPts.length;j++) {
                if (assigned[j]) continue;
                const dx=allPts[j].x-allPts[i].x, dy=allPts[j].y-allPts[i].y;
                if (dx*dx+dy*dy<=MERGE_R2) {
                  assigned[j]=1;
                  const u=allPts[j].u;
                  cl.us.push(u);cl.sx+=allPts[j].x;cl.sy+=allPts[j].y;cl.n++;
                  cl.lats.push(u[13]);cl.lons.push(u[14]);cl.ct[u[4]]++;
                }
              }
              clusters.push({
                x:cl.sx/cl.n, y:cl.sy/cl.n, count:cl.n, us:cl.us,
                minLat:Math.min(...cl.lats), maxLat:Math.max(...cl.lats),
                minLon:Math.min(...cl.lons), maxLon:Math.max(...cl.lons),
                domCat:cl.ct.indexOf(Math.max(...cl.ct)),
                single:cl.n===1?cl.us[0]:null,
              });
            }
          }

          // Zoom helpers
          const zoomIn = () => {
            const cLa=(b.minLat+b.maxLat)/2, cLo=(b.minLon+b.maxLon)/2;
            const dLa=(b.maxLat-b.minLat)*0.28, dLo=(b.maxLon-b.minLon)*0.28;
            setMapView({minLat:cLa-dLa,maxLat:cLa+dLa,minLon:cLo-dLo,maxLon:cLo+dLo});
            setMapSelCo(null); setMapCluster(null);
          };
          const zoomOut = () => {
            const cLa=(b.minLat+b.maxLat)/2, cLo=(b.minLon+b.maxLon)/2;
            const dLa=Math.min((b.maxLat-b.minLat)/0.52,(TG_FULL.maxLat-TG_FULL.minLat)*0.58);
            const dLo=Math.min((b.maxLon-b.minLon)/0.52,(TG_FULL.maxLon-TG_FULL.minLon)*0.58);
            setMapView({
              minLat:Math.max(TG_FULL.minLat-0.3,cLa-dLa), maxLat:Math.min(TG_FULL.maxLat+0.3,cLa+dLa),
              minLon:Math.max(TG_FULL.minLon-0.3,cLo-dLo), maxLon:Math.min(TG_FULL.maxLon+0.3,cLo+dLo),
            });
            setMapSelCo(null); setMapCluster(null);
          };
          const zoomToCluster = (cl) => {
            const geoSpan = Math.max(cl.maxLat-cl.minLat, cl.maxLon-cl.minLon);
            if (geoSpan < 0.008) {
              // Companies at same/very close GPS ? can't separate by zooming
              // Show cluster detail panel instead
              setMapCluster(cl);
              setMapSelCo(null);
              return;
            }
            const padLa=Math.max(0.04,(cl.maxLat-cl.minLat)*0.5);
            const padLo=Math.max(0.04,(cl.maxLon-cl.minLon)*0.5);
            setMapView({minLat:cl.minLat-padLa,maxLat:cl.maxLat+padLa,
                        minLon:cl.minLon-padLo,maxLon:cl.maxLon+padLo});
            setMapSelCo(null); setMapCluster(null);
          };

          const singletons = SHOW_ALL ? clusters.length : clusters.filter(c=>c.count===1).length;

          return (
            <>
              {/* Controls */}
              <div style={{display:"flex",gap:8,marginBottom:10,alignItems:"center",flexWrap:"wrap"}}>
                <button onClick={()=>{setMapView({...TG_FULL});setMapSelCo(null);setMapCluster(null);}} style={C.sel}>Reset</button>
                <button onClick={zoomOut} style={C.sel}>- Zoom out</button>
                <button onClick={zoomIn} style={C.sel}>+ Zoom in</button>
                <span style={{fontSize:10,color:"#666"}}>
                  {SHOW_ALL
                    ? `${clusters.length} companies (individual view)`
                    : `${clusters.length} clusters - zoom in for mandal detail`}
                </span>
                <span style={{fontSize:10,color:"#aaa",marginLeft:"auto"}}>
                  {lonSpan > 2 ? "State" : lonSpan > 0.5 ? "District" : lonSpan > 0.1 ? "Mandal" : "Company"} view
                </span>
              </div>

              <div style={{display:"flex",gap:12}}>

                {/* MAP SVG */}
                <div style={{...C.card,padding:0,overflow:"hidden",flex:1,background:"#0d3349"}}>
                  <svg viewBox={`0 0 ${SW} ${SH}`} style={{width:"100%",display:"block",cursor:"crosshair"}}
                    onClick={()=>{setMapSelCo(null);setMapCluster(null);}}>

                    {/* Clip path = exact Telangana outline computed via shapely union of all districts.
                        Everything outside is masked out ? no stray dots, clean border. */}
                    <defs>
                      <clipPath id="tg-clip">
                        <path d={TG_OUTLINE.map((p,i)=>{
                          const q=proj(p[0],p[1]);
                          return `${i===0?'M':'L'}${q.x.toFixed(1)},${q.y.toFixed(1)}`;
                        }).join(' ')+'Z'}/>
                      </clipPath>
                    </defs>

                    {/* Surrounding states ? muted grey, distinct from Telangana */}
                    <rect width={SW} height={SH} fill="#9eaab5"/>

                    {/* Telangana base fill ? covers any sub-pixel gaps between district polygons */}
                    <path d={outlinePath} fill="#c8d4a0" stroke="none"/>

                    {/* Everything inside Telangana ? clipped so nothing bleeds outside */}
                    <g clipPath="url(#tg-clip)">

                    {/* District fills ? each district its own pastel color.
                        Stroke same color as fill to seal any sub-pixel gaps at shared edges. */}
                    {(()=>{
                      const DIST_COL = {
                        "Adilabad":          "#f4a3a3",
                        "Bhadradri":         "#a3cff4",
                        "Hydrabad":          "#f4d3a3",
                        "Jagtial":           "#b3f4a3",
                        "Jangaon":           "#d4a3f4",
                        "Jayashankar":       "#a3f4d4",
                        "Jogulamba":         "#f4f4a3",
                        "Kamareddy":         "#f4b8d4",
                        "Karimnagar":        "#a3d4f4",
                        "Khammam":           "#c4f4a3",
                        "Komaram Bheem":     "#f4c4a3",
                        "Mahabubabad":       "#b8a3f4",
                        "Mahabubnagar":      "#a3f4b8",
                        "Mancherial":        "#f4a3c4",
                        "Medak":             "#a3f4f4",
                        "Medchal":           "#f4d4a3",
                        "Mulugu":            "#c4a3f4",
                        "Nagarkurnool":      "#a3c4f4",
                        "Nalgonda":          "#f4a3b8",
                        "Narayanpet":        "#b4f4a3",
                        "Nirmal":            "#f4b4a3",
                        "Nizamabad":         "#a3b4f4",
                        "Peddapalli":        "#d4f4a3",
                        "Rajanna Sircilla":  "#f4a3d4",
                        "Rangareddy":        "#a3f4c4",
                        "Sangareddy":        "#f4c4b4",
                        "Siddipet":          "#b4a3f4",
                        "Suryapet":          "#a3f4a3",
                        "Vikarabad":         "#f4b4c4",
                        "Wanaparthy":        "#c4f4b4",
                        "Warangal (R)":      "#a3c8f4",
                        "Warangal (U)":      "#f4d8a3",
                        "Yadadri Bhongiri":  "#d4f4c4",
                      };
                      return Object.entries(TG_DISTRICTS).map(([name,poly]) => {
                        const col = DIST_COL[name] || "#d0d0d0";
                        return (
                          <path key={`f${name}`} d={polyPath(poly)}
                            fill={col} stroke={col} strokeWidth={1.5}/>
                        );
                      });
                    })()}

                    {/* District boundary lines on top ? white strokes, clipped to Telangana */}
                    {Object.entries(TG_DISTRICTS).map(([name,poly]) => (
                      <path key={`s${name}`} d={polyPath(poly)} fill="none"
                        stroke="rgba(80,80,80,0.7)" strokeWidth={0.8}/>
                    ))}

                    {/* Mandal boundaries ? shown when zoomed to district/mandal level */}
                    {lonSpan < 2.5 && TG_MANDALS.map((m,i) => {
                      // Quick bbox check before rendering
                      const lats=m.poly.map(p=>p[0]), lons=m.poly.map(p=>p[1]);
                      const mxLa=Math.max(...lats),mnLa=Math.min(...lats);
                      const mxLo=Math.max(...lons),mnLo=Math.min(...lons);
                      if (mxLa<b.minLat||mnLa>b.maxLat||mxLo<b.minLon||mnLo>b.maxLon) return null;
                      return (
                        <path key={i} d={polyPath(m.poly)}
                          fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth={0.4}/>
                      );
                    })}

                    {/* District labels ? visible at state + district zoom */}
                    {/* DOTS — rendered before labels so labels sit on top */}
                    {clusters.map((cl,i) => {
                      if (cl.single) {
                        // Individual company dot
                        const u=cl.single;
                        const isSel=mapSelCo?.[0]===u[0];
                        const r=isSel?8:5;
                        return (
                          <g key={i} style={{cursor:"pointer"}}
                            onClick={e=>{e.stopPropagation();setMapSelCo(u);setMapCluster(null);}}>
                            {isSel && <circle cx={cl.x} cy={cl.y} r={r+5} fill={CAT_COL[u[4]]} opacity={0.2}/>}
                            <circle cx={cl.x} cy={cl.y} r={r+1.5} fill="none" stroke={PCB_COL[u[5]]} strokeWidth={1.5} opacity={0.8}/>
                            <circle cx={cl.x} cy={cl.y} r={r} fill={CAT_COL[u[4]]}
                              stroke={isSel?"#fff":"rgba(255,255,255,0.7)"} strokeWidth={isSel?1.5:0.6} opacity={0.95}/>
                          </g>
                        );
                      }
                      // Cluster bubble — count shown only when 2+ companies
                      const label = cl.count>=10000 ? Math.round(cl.count/1000)+"k"
                                  : cl.count>=1000  ? (cl.count/1000).toFixed(1)+"k"
                                  : String(cl.count);
                      const r = Math.max(12, Math.min(22, 10+label.length*3.5));
                      const col = CAT_COL[cl.domCat];
                      const isSelCl = mapCluster===cl;
                      return (
                        <g key={i} style={{cursor:"pointer"}}
                          onClick={e=>{e.stopPropagation();zoomToCluster(cl);}}>
                          <circle cx={cl.x} cy={cl.y} r={r+2} fill="rgba(0,0,0,0.25)"/>
                          <circle cx={cl.x} cy={cl.y} r={r} fill={col}
                            stroke={isSelCl?"#fff":"rgba(255,255,255,0.85)"} strokeWidth={isSelCl?2.5:1.5} opacity={0.93}/>
                          {cl.count > 1 && <text x={cl.x} y={cl.y+0.5} textAnchor="middle" dominantBaseline="middle"
                            fontSize={r>16?9:8} fill="#fff" fontWeight={700} pointerEvents="none">
                            {label}
                          </text>}
                        </g>
                      );
                    })}

                    {/* District labels — rendered AFTER dots so always visible on top */}
                    {lonSpan > 0.5 && Object.entries(TG_DISTRICTS).map(([name,poly]) => {
                      const lats=poly.map(p=>p[0]), lons=poly.map(p=>p[1]);
                      const cLa=(Math.min(...lats)+Math.max(...lats))/2;
                      const cLo=(Math.min(...lons)+Math.max(...lons))/2;
                      const pt=proj(cLa,cLo);
                      if (pt.x<6||pt.x>SW-6||pt.y<6||pt.y>SH-6) return null;
                      const label = name
                        .replace(' (R)','').replace(' (U)','')
                        .replace('Yadadri Bhongiri','Yadadri')
                        .replace('Komaram Bheem','Komaram\nBheem')
                        .replace('Rajanna Sircilla','Rajanna\nSircilla')
                        .replace('Rangareddy','Ranga\nReddy')
                        .replace('Mahabubnagar','Mahbub\nnagar')
                        .replace('Mahabubabad','Mahbub\nabad')
                        .replace('Nagarkurnool','Nagarku\nrnool')
                        .trim();
                      const fs = lonSpan > 3 ? 8 : lonSpan > 1.5 ? 10 : 12;
                      const lines = label.split('\n');
                      const lineH = fs * 1.3;
                      const maxLen = Math.max(...lines.map(l=>l.length));
                      const pw = maxLen * fs * 0.58 + 8;
                      const ph = lines.length * lineH + 5;
                      return (
                        <g key={name} pointerEvents="none" transform={`translate(${pt.x},${pt.y})`}>
                          <rect x={-pw/2} y={-ph/2} width={pw} height={ph}
                            rx={3} fill="rgba(255,255,255,0.85)" stroke="rgba(0,0,0,0.2)" strokeWidth={0.5}/>
                          {lines.map((line,li) => (
                            <text key={li} x={0} y={(li-(lines.length-1)/2)*lineH+0.5}
                              textAnchor="middle" dominantBaseline="middle"
                              fontSize={fs} fill="#111" fontWeight={700}>{line}</text>
                          ))}
                        </g>
                      );
                    })}

                    {/* Mandal labels — white pill, shown at mandal zoom, rendered after dots */}
                    {lonSpan < 0.8 && lonSpan > 0.12 && TG_MANDALS.map((m,i) => {
                      const aLa=m.poly.reduce((s,p)=>s+p[0],0)/m.poly.length;
                      const aLo=m.poly.reduce((s,p)=>s+p[1],0)/m.poly.length;
                      if (aLa<b.minLat||aLa>b.maxLat||aLo<b.minLon||aLo>b.maxLon) return null;
                      const pt=proj(aLa,aLo);
                      if (pt.x<4||pt.x>SW-4||pt.y<4||pt.y>SH-4) return null;
                      const fs = lonSpan < 0.3 ? 8.5 : 7;
                      const pw = m.name.length * fs * 0.55 + 6;
                      const ph = fs * 1.4 + 3;
                      return (
                        <g key={i} pointerEvents="none" transform={`translate(${pt.x},${pt.y})`}>
                          <rect x={-pw/2} y={-ph/2} width={pw} height={ph}
                            rx={2} fill="rgba(255,255,255,0.82)" stroke="rgba(0,0,0,0.15)" strokeWidth={0.4}/>
                          <text x={0} y={0.5} textAnchor="middle" dominantBaseline="middle"
                            fontSize={fs} fill="#222" fontWeight={600}>{m.name}</text>
                        </g>
                      );
                    })}

                    </g>{/* end clip group */}

                    {/* Outer Telangana border — drawn outside clip so it's always crisp on top */}
                    <path d={outlinePath} fill="none" stroke="#333" strokeWidth={2}/>

                    {/* Compass ? outside clip so always visible */}
                    <g transform={`translate(${SW-24},${SH-24})`} opacity={0.55} pointerEvents="none">
                      <circle r={12} fill="rgba(255,255,255,0.12)" stroke="rgba(255,255,255,0.35)" strokeWidth={0.5}/>
                      <text textAnchor="middle" y={-3} fontSize={7} fill="#fff" fontWeight={700}>N</text>
                      <line y1={-8} y2={8} stroke="rgba(255,255,255,0.5)" strokeWidth={0.8}/>
                      <line x1={-8} x2={8} stroke="rgba(255,255,255,0.5)" strokeWidth={0.8}/>
                    </g>

                    {/* Legend */}
                    <g transform="translate(8,8)" pointerEvents="none">
                      <rect x={-3} y={-3} width={108} height={58} rx={4} fill="rgba(0,0,0,0.5)"/>
                      <text x={0} y={10} fontSize={7.5} fill="#fff" fontWeight={600}>Company size</text>
                      {CATS.slice(0,4).map((cat,i)=>(
                        <g key={cat} transform={`translate(0,${14+i*11})`}>
                          <circle cx={5} cy={5} r={4} fill={CAT_COL[i]}/>
                          <text x={13} y={9} fontSize={7.5} fill="rgba(255,255,255,0.85)">{cat}</text>
                        </g>
                      ))}
                    </g>
                  </svg>
                </div>

                {/* SIDE PANEL */}
                <div style={{width:230,flexShrink:0,display:"flex",flexDirection:"column",gap:10}}>

                  {/* Company detail */}
                  {mapSelCo && (
                    <div style={C.card}>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
                        <div style={{fontSize:12,fontWeight:600,lineHeight:1.35,flex:1,paddingRight:4}}>{mapSelCo[1]}</div>
                        <button onClick={()=>setMapSelCo(null)} style={{background:"none",border:"none",fontSize:17,cursor:"pointer",color:"#888",padding:0}}>&times;</button>
                      </div>
                      <div style={{fontSize:11,color:"#888",marginBottom:8}}>{districts[mapSelCo[2]]} · {mandals[mapSelCo[3]]}</div>
                      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:5,marginBottom:10}}>
                        {[
                          ["Size",<span style={C.badge(CAT_COL[mapSelCo[4]])}>{CATS[mapSelCo[4]]}</span>],
                          ["PCB",<span style={C.badge(PCB_COL[mapSelCo[5]])}>{PCBS[mapSelCo[5]]}</span>],
                          ["Investment",fmt(mapSelCo[7])],
                          ["Employment",fmtE(mapSelCo[8])],
                          ["Turnover",fmt(mapSelCo[9])],
                          ["Estate",mapSelCo[10]?"Yes":"No"],
                          ["Women-led",mapSelCo[11]?"Yes":"No"],
                          ["ODOP",mapSelCo[12]?"Yes":"No"],
                        ].map(([l,v])=>(
                          <div key={l} style={C.kpi}>
                            <div style={{fontSize:10,color:"#888"}}>{l}</div>
                            <div style={{fontSize:11,fontWeight:500}}>{v}</div>
                          </div>
                        ))}
                      </div>
                      <button onClick={()=>openNetwork(mapSelCo)} style={{...C.aiBtn,fontSize:11,width:"100%"}}>Full network &rarr;</button>
                    </div>
                  )}

                  {/* Cluster detail panel ? all companies when GPS can't separate them */}
                  {mapCluster && (
                    <div style={{...C.card,flex:1,overflow:"hidden",display:"flex",flexDirection:"column"}}>
                      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:8}}>
                        <div>
                          <div style={{fontSize:12,fontWeight:700}}>{mapCluster.count} companies</div>
                          <div style={{fontSize:10,color:"#888"}}>at same location — all listed below</div>
                        </div>
                        <button onClick={()=>setMapCluster(null)} style={{background:"none",border:"none",fontSize:17,cursor:"pointer",color:"#888",padding:0}}>&times;</button>
                      </div>
                      <div style={{overflowY:"auto",flex:1,maxHeight:420}}>
                        {mapCluster.us.map((u,i)=>(
                          <div key={i} style={{padding:"6px 8px",borderBottom:"1px solid #f0f0f0",cursor:"pointer",
                            borderLeft:`3px solid ${CAT_COL[u[4]]}`}}
                            onClick={()=>{setMapSelCo(u);setMapCluster(null);}}>
                            <div style={{fontSize:11,fontWeight:500,lineHeight:1.3}}>{u[1]}</div>
                            <div style={{fontSize:10,color:"#888",marginTop:2}}>
                              {CATS[u[4]]} · {districts[u[2]]} · {mandals[u[3]]}
                            </div>
                            <div style={{fontSize:10,color:"#555",marginTop:1}}>
                              Inv: {fmt(u[7])} · Emp: {fmtE(u[8])}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Default info panel */}
                  {!mapSelCo && !mapCluster && (
                    <div style={C.card}>
                      <div style={{fontSize:12,fontWeight:700,marginBottom:8,color:"#0C4E64"}}>Telangana MSME Map</div>
                      <div style={{fontSize:22,fontWeight:700,color:"#1e7dc1"}}>{geoUnits.length.toLocaleString()}</div>
                      <div style={{fontSize:11,color:"#888",marginBottom:8}}>companies with GPS data</div>
                      <div style={{fontSize:18,fontWeight:700}}>{clusters.length}</div>
                      <div style={{fontSize:11,color:"#888",marginBottom:10}}>
                        {SHOW_ALL ? "individual companies in view" : "clusters in view"}
                      </div>
                      <div style={{fontSize:11,color:"#555",lineHeight:1.9}}>
                        {SHOW_ALL ? (
                          <>
                            <p>- Each dot = one company, exact GPS</p>
                            <p>- Dot fill = size, ring = PCB risk</p>
                            <p>- Click dot for full details</p>
                          </>
                        ) : (
                          <>
                            <p>- Click cluster to zoom in</p>
<p>- If companies share GPS, panel lists all of them</p>
                            <p>- Mandal boundaries appear below district zoom</p>
                            <p>- All {geoUnits.length.toLocaleString()} companies covered</p>
                          </>
                        )}
                      </div>
                      <div style={{marginTop:10,display:"flex",flexDirection:"column",gap:4}}>
                        {CATS.slice(0,4).map((cat,i)=>(
                          <div key={cat} style={{display:"flex",alignItems:"center",gap:6}}>
                            <div style={{width:10,height:10,borderRadius:"50%",background:CAT_COL[i]}}/>
                            <span style={{fontSize:11}}>{cat}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

              </div>
            </>
          );
        })()}

        {tab === "network" && (
          <>
            {/* Network search bar */}
            <div style={C.card}>
              <div style={{fontSize:12, fontWeight:500, marginBottom:8}}>Select a company to view its supply chain network</div>
              <div style={{display:"flex", gap:8}}>
                <input
                  placeholder="Type company name..."
                  style={{...C.aiInput, padding:"8px 12px", fontSize:12}}
                  onChange={e => {
                    const v = e.target.value.toLowerCase().trim();
                    if (v.length < 3) return;
                    const found = units.find(u => u[1].toLowerCase().includes(v));
                    if (found) openNetwork(found);
                  }}
                />
                {networkTarget && (
                  <div style={{fontSize:12, color:"#888", alignSelf:"center", whiteSpace:"nowrap"}}>
                    Showing: <strong style={{color:"#1a1a1a"}}>{networkTarget[1]}</strong>
                  </div>
                )}
              </div>
            </div>

            {networkTarget ? (
              <div style={{display:"grid", gridTemplateColumns:"1fr 380px", gap:14}}>
                {/* Force-directed graph */}
                <div style={C.card}>
                  <div style={{fontSize:11, fontWeight:500, color:"#888", textTransform:"uppercase", letterSpacing:"0.5px", marginBottom:8}}>
                    Supply chain network - {networkTarget[1]}
                  </div>
                  <svg viewBox="0 0 600 450" style={{width:"100%", borderRadius:8, border:"1px solid #e5e5e5"}}>
                    <defs>
                      <marker id="arrowS" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#1D9E75" opacity="0.7"/></marker>
                      <marker id="arrowB" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#378ADD" opacity="0.7"/></marker>
                      <marker id="arrowR" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#E24B4A" opacity="0.7"/></marker>
                    </defs>
                    {/* Links */}
                    {networkData.links.map((l, i) => {
                      const s = networkPositions.find(p => p.id === l.source);
                      const t = networkPositions.find(p => p.id === l.target);
                      if (!s || !t) return null;
                      const col = l.type==="supplier"?"#1D9E75":l.type==="buyer"?"#378ADD":l.type==="risk"?"#E24B4A":"#888";
                      const marker = l.type==="supplier"?"url(#arrowS)":l.type==="buyer"?"url(#arrowB)":l.type==="risk"?"url(#arrowR)":"";
                      const dash = l.type==="risk"?"4,3":l.type==="peer"?"2,4":"none";
                      const isSel = selectedNetLink && selectedNetLink._i === i;
                      return (
                        <g key={i} style={{cursor:"pointer"}} onClick={(e)=>{e.stopPropagation();setSelectedNetLink({...l, _i:i, _srcNode:s, _tgtNode:t});}}>
                          <line x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke="transparent" strokeWidth={12}/>
                          <line x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke={isSel?"#1a1a1a":col} strokeWidth={isSel?4:l.type==="risk"?2:1.5} opacity={isSel?1:0.6} strokeDasharray={isSel?"none":dash} markerEnd={isSel?"":marker}/>
                          {isSel && <>
                            <circle cx={s.x} cy={s.y} r={6} fill={col} stroke="#fff" strokeWidth={2}/>
                            <circle cx={t.x} cy={t.y} r={6} fill={col} stroke="#fff" strokeWidth={2}/>
                          </>}
                        </g>
                      );
                    })}
                    {/* Nodes */}
                    {networkPositions.map(n => {
                      const r = n.role==="target" ? 18 : Math.max(7, Math.min(15, Math.sqrt(n.investment||1)*0.15));
                      const col = CAT_COL[n.catIdx] || "#888";
                      const pcbStroke = PCB_COL[n.pcbIdx] || "#888";
                      return (
                        <g key={n.id} style={{cursor:"pointer"}} onClick={()=>{const u=units.find(x=>x[0]===n.id);if(u)openNetwork(u);}}>
                          <circle cx={n.x} cy={n.y} r={r+2} fill="none" stroke={pcbStroke} strokeWidth={2} opacity={0.5}/>
                          <circle cx={n.x} cy={n.y} r={r} fill={col} opacity={n.role==="target"?1:0.8} stroke="#fff" strokeWidth={1}/>
                          <text x={n.x} y={n.y-r-6} textAnchor="middle" fontSize={n.role==="target"?10:8} fill="#1a1a1a" fontWeight={n.role==="target"?600:400}>
                            {n.name.length > 18 ? n.name.slice(0,16)+"..." : n.name}
                          </text>
                          {n.role==="target" && <text x={n.x} y={n.y+4} textAnchor="middle" fontSize={7} fill="#fff" fontWeight={600}>{CATS[n.catIdx]}</text>}
                        </g>
                      );
                    })}
                    {/* Legend */}
                    <g transform="translate(10,410)">
                      {[["#1D9E75","Supplier"],["#378ADD","Buyer"],["#E24B4A","Risk link"],["#888","Peer"]].map(([c,l],i)=>(
                        <g key={l} transform={`translate(${i*130},0)`}>
                          <line x1={0} y1={5} x2={20} y2={5} stroke={c} strokeWidth={2}/>
                          <text x={24} y={9} fontSize={9} fill="#888">{l}</text>
                        </g>
                      ))}
                    </g>
                    <g transform="translate(10,425)">
                      {CATS.map((c,i)=>(
                        <g key={c} transform={`translate(${i*100},0)`}>
                          <circle cx={6} cy={5} r={5} fill={CAT_COL[i]}/>
                          <text x={16} y={9} fontSize={9} fill="#888">{c}</text>
                        </g>
                      ))}
                    </g>
                  </svg>
                  <div style={{fontSize:10, color:"#888", marginTop:6}}>
                    Node size = investment . Outer ring = PCB tier . Click node to recenter . Click link line to see why
                  </div>
                </div>

                {/* Right panel: company detail + similar */}
                <div>
                  {/* Target company detail */}
                  <div style={C.card}>
                    <div style={{fontSize:15, fontWeight:600, marginBottom:8}}>{networkTarget[1]}</div>
                    <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, marginBottom:12}}>
                      {[
                        ["District", districts[networkTarget[2]]||"--"],
                        ["Mandal", mandals[networkTarget[3]]||"--"],
                        ["Category", CATS[networkTarget[4]]],
                        ["PCB", PCBS[networkTarget[5]]+" - "+PCB_LABEL[networkTarget[5]]],
                        ["Investment", fmt(networkTarget[7])],
                        ["Employment", fmtE(networkTarget[8])],
                        ["Turnover", fmt(networkTarget[9])],
                        ["Estate", networkTarget[10]?"In estate":"Outside"],
                      ].map(([l,v])=>(
                        <div key={l} style={{...C.kpi, padding:"8px 10px"}}>
                          <div style={{fontSize:9, color:"#888"}}>{l}</div>
                          <div style={{fontSize:12, fontWeight:500}}>{v}</div>
                        </div>
                      ))}
                    </div>
                    <div style={{display:"flex",gap:6}}>
                      <button onClick={()=>{setSelected(networkTarget);setTab("search");}} style={{...C.sel, fontSize:11, flex:1}}>View profile</button>
                      <button onClick={()=>{const qry=`Find suppliers for ${networkTarget[1]}`;setAiQuery(qry);runQuery(qry);}} style={{...C.sel, fontSize:11, flex:1, background:"#0C4E64", color:"#fff", border:"none"}}>Find suppliers</button>
                    </div>
                  </div>

                  {/* Network stats */}
                  <div style={C.card}>
                    <div style={{fontSize:11, fontWeight:500, color:"#888", textTransform:"uppercase", marginBottom:8}}>Network composition</div>
                    <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:6}}>
                      <div style={C.kpi}><div style={C.kpiV}>{networkData.nodes.filter(n=>n.role==="supplier").length}</div><div style={C.kpiL}>Suppliers</div></div>
                      <div style={C.kpi}><div style={C.kpiV}>{networkData.nodes.filter(n=>n.role==="buyer").length}</div><div style={C.kpiL}>Buyers</div></div>
                      <div style={C.kpi}><div style={C.kpiV}>{networkData.nodes.filter(n=>n.role==="peer").length}</div><div style={C.kpiL}>Mandal peers</div></div>
                      <div style={C.kpi}><div style={{...C.kpiV, color:networkData.links.filter(l=>l.type==="risk").length?"#E24B4A":"#1D9E75"}}>{networkData.links.filter(l=>l.type==="risk").length}</div><div style={C.kpiL}>Risk links</div></div>
                    </div>
                  </div>

                  {/* Link explanation */}
                  {selectedNetLink && (() => {
                    const l = selectedNetLink;
                    const srcNode = l._srcNode;
                    const tgtNode = l._tgtNode;
                    const srcUnit = units.find(u => u[0] === l.source);
                    const tgtUnit = units.find(u => u[0] === l.target);
                    const srcName = srcNode?.name || "Unknown";
                    const tgtName = tgtNode?.name || "Unknown";
                    const srcCat = srcNode ? CATS[srcNode.catIdx] : "?";
                    const tgtCat = tgtNode ? CATS[tgtNode.catIdx] : "?";

                    const srcSec = srcUnit ? getSec(srcUnit) : null;
                    const tgtSec = tgtUnit ? getSec(tgtUnit) : null;
                    const targetSec = getSec(networkTarget);
                    const CHAIN_WHY_SHORT = {
                      "chemical->health":"supplies APIs, excipients, raw materials for pharma",
                      "chemical->textile":"supplies dyes, finishing agents, processing chemicals",
                      "chemical->food":"supplies preservatives, additives, processing agents",
                      "chemical->construction":"supplies paints, coatings, adhesives",
                      "chemical->auto":"supplies lubricants, coatings, polymer parts",
                      "packaging->food":"supplies containers, wraps, labels",
                      "packaging->health":"supplies blister packs, vials, cartons",
                      "packaging->chemical":"supplies drums, containers",
                      "construction->auto":"supplies steel, castings, structural components",
                      "auto->construction":"supplies machinery, equipment, tools",
                      "energy->auto":"supplies batteries, power systems, EV components",
                      "furniture->construction":"supplies interior fittings, doors, cabinets",
                      "construction->energy":"supplies infrastructure for power projects",
                    };

                    const reasons = [];
                    if (l.type === "supplier") {
                      reasons.push({icon: "1", color: "#1D9E75", text: `${srcName} is a potential supplier to ${networkTarget[1]}`});
                      reasons.push({icon: "2", color: "#1D9E75", text: `Size hierarchy: ${srcCat} (smaller) supplies to ${CATS[networkTarget[4]]} (larger) - smaller companies feed larger ones`});
                      if (srcUnit && srcUnit[2] === networkTarget[2]) reasons.push({icon: "3", color: "#378ADD", text: `Same district: ${districts[networkTarget[2]]} - close enough for regular supply`});
                      if (srcUnit && srcUnit[3] === networkTarget[3]) reasons.push({icon: "3", color: "#7F77DD", text: `Same mandal: ${mandals[networkTarget[3]]} - strong geographic proximity`});
                      if (srcSec && targetSec && srcSec !== targetSec) {
                        const why = CHAIN_WHY_SHORT[`${srcSec}->${targetSec}`];
                        if (why) reasons.push({icon: "4", color: "#D85A30", text: `Sector link: ${srcSec} ${why}`});
                        else reasons.push({icon: "4", color: "#D85A30", text: `Cross-sector: ${srcSec} sector can provide inputs to ${targetSec} sector`});
                      }
                      if (srcSec && targetSec && srcSec === targetSec) reasons.push({icon: "4", color: "#EF9F27", text: `Same sector (${srcSec}) - could be a sub-contractor or component supplier`});
                      if (!srcSec || !targetSec) reasons.push({icon: "4", color: "#888", text: "Sector not detected from company name - link based on size + geography only"});
                    } else if (l.type === "buyer") {
                      reasons.push({icon: "1", color: "#378ADD", text: `${tgtName} is a potential buyer from ${networkTarget[1]}`});
                      reasons.push({icon: "2", color: "#378ADD", text: `Size hierarchy: ${CATS[networkTarget[4]]} (smaller) sells to ${tgtCat} (larger)`});
                      if (tgtUnit && tgtUnit[2] === networkTarget[2]) reasons.push({icon: "3", color: "#378ADD", text: `Same district: ${districts[networkTarget[2]]}`});
                      if (tgtSec && targetSec && tgtSec !== targetSec) {
                        const why = CHAIN_WHY_SHORT[`${targetSec}->${tgtSec}`];
                        if (why) reasons.push({icon: "4", color: "#D85A30", text: `Sector link: ${targetSec} ${why}`});
                        else reasons.push({icon: "4", color: "#D85A30", text: `Cross-sector: ${targetSec} can supply inputs to ${tgtSec}`});
                      }
                    } else if (l.type === "risk") {
                      const riskNode = networkPositions.find(p => p.id === l.source);
                      reasons.push({icon: "!", color: "#E24B4A", text: "Supply chain risk link"});
                      if (riskNode && riskNode.pcbIdx === 2) reasons.push({icon: "!", color: "#E24B4A", text: `${riskNode.name} has RED PCB tier (high pollution)`});
                      const riskUnit = units.find(u => u[0] === l.source);
                      if (riskUnit && riskUnit[10] === 0) reasons.push({icon: "!", color: "#E24B4A", text: "Located outside industrial estate (compliance risk)"});
                      reasons.push({icon: "?", color: "#EF9F27", text: "If this supplier faces regulatory action, it could disrupt the supply chain"});
                    } else if (l.type === "peer") {
                      reasons.push({icon: "=", color: "#888", text: `Both operate in the same mandal: ${mandals[networkTarget[3]]}`});
                      reasons.push({icon: "=", color: "#888", text: "Mandal peers share infrastructure, labor pool, and logistics"});
                      reasons.push({icon: "?", color: "#EF9F27", text: "Could be competitors, collaborators, or part of an informal cluster"});
                    }

                    return (
                      <div style={{...C.card, borderLeft:`3px solid ${l.type==="supplier"?"#1D9E75":l.type==="buyer"?"#378ADD":l.type==="risk"?"#E24B4A":"#888"}`, position:"relative"}}>
                        <button onClick={()=>setSelectedNetLink(null)} style={{position:"absolute",top:8,right:10,background:"none",border:"none",fontSize:16,cursor:"pointer",color:"#888"}}>x</button>
                        <div style={{fontSize:11,fontWeight:500,color:l.type==="supplier"?"#1D9E75":l.type==="buyer"?"#378ADD":l.type==="risk"?"#E24B4A":"#888",textTransform:"uppercase",marginBottom:10}}>
                          Why this {l.type} link exists
                        </div>

                        <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}>
                          <div style={{flex:1,background:"#fafafa",borderRadius:6,padding:"8px 10px",border:"1px solid #e5e5e5"}}>
                            <div style={{fontSize:10,color:"#888"}}>{l.type==="buyer"?"From":"Supplier"}</div>
                            <div style={{fontSize:12,fontWeight:500}}>{srcName.slice(0,25)}</div>
                            <div style={{fontSize:10,color:"#888"}}>{srcCat}</div>
                          </div>
                          <div style={{fontSize:14,color:"#888"}}>{l.type==="risk"?"!!":"-->"}</div>
                          <div style={{flex:1,background:"#fafafa",borderRadius:6,padding:"8px 10px",border:"1px solid #e5e5e5"}}>
                            <div style={{fontSize:10,color:"#888"}}>{l.type==="buyer"?"Buyer":"To"}</div>
                            <div style={{fontSize:12,fontWeight:500}}>{tgtName.slice(0,25)}</div>
                            <div style={{fontSize:10,color:"#888"}}>{tgtCat}</div>
                          </div>
                        </div>

                        {reasons.map((r, ri) => (
                          <div key={ri} style={{display:"flex",gap:8,marginBottom:6,fontSize:12,lineHeight:1.5}}>
                            <div style={{width:18,height:18,borderRadius:9,background:r.color+"22",color:r.color,fontSize:10,fontWeight:600,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0}}>{r.icon}</div>
                            <div style={{color:"#555"}}>{r.text}</div>
                          </div>
                        ))}
                      </div>
                    );
                  })()}

                  {/* Similar companies (matching engine) */}
                  {similarResults.length > 0 && (
                    <div style={C.card}>
                      <div style={{fontSize:11, fontWeight:500, color:"#888", textTransform:"uppercase", marginBottom:8}}>
                        Most similar companies (by investment, size, location)
                      </div>
                      {similarResults.map(({unit: s, score}) => (
                        <div key={s[0]} style={{...C.linkCard("buyer"), cursor:"pointer"}} onClick={()=>openNetwork(s)}>
                          <div style={{display:"flex", justifyContent:"space-between", alignItems:"center"}}>
                            <div>
                              <div style={{fontSize:12, fontWeight:500}}>{s[1]}</div>
                              <div style={{fontSize:11, color:"#888", marginTop:1}}>
                                {districts[s[2]]||"--"} . {CATS[s[4]]} . {fmt(s[7])} . {fmtE(s[8])} emp
                              </div>
                            </div>
                            <div style={{fontSize:10, padding:"2px 8px", borderRadius:10, background:`rgba(13,78,100,${score})`, color:score>0.5?"#fff":"#0C4E64", fontWeight:500}}>
                              {(score*100).toFixed(0)}%
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div style={{...C.card, textAlign:"center", padding:48, color:"#888"}}>
                <div style={{fontSize:28, marginBottom:12}}>⬡</div>
                <div style={{fontSize:14, fontWeight:500, marginBottom:6}}>Supply chain network explorer</div>
                <div style={{fontSize:12, marginBottom:16}}>Click any company or search above to visualize its buyer-supplier network</div>
                <div style={{display:"flex", flexWrap:"wrap", gap:6, justifyContent:"center"}}>
                  {units.filter(u=>u[4]>=3).sort((a,b)=>b[7]-a[7]).slice(0,12).map(u=>(
                    <button key={u[0]} style={C.chip()} onClick={()=>openNetwork(u)}>
                      {u[1].length > 25 ? u[1].slice(0,23)+"..." : u[1]}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* ?????? SUPPLY LINKS TAB ?????? */}
        {tab === "links" && (() => {
          const mega = units.filter(u=>u[4]===4).sort((a,b)=>b[7]-a[7]);
          const large = units.filter(u=>u[4]===3).sort((a,b)=>b[7]-a[7]).slice(0,20);
          const redRisk = units.filter(u=>u[5]===2&&u[10]===0).sort((a,b)=>b[7]-a[7]).slice(0,15);
          const womenLarge = units.filter(u=>u[11]===1&&u[4]>=2).sort((a,b)=>b[7]-a[7]).slice(0,15);
          const odopBig = units.filter(u=>u[12]===1).sort((a,b)=>b[7]-a[7]).slice(0,15);

          const Section = ({title, data, color="#0C4E64", note}) => (
            <div style={C.card}>
              <div style={{fontSize:12,fontWeight:500,color:"#888",textTransform:"uppercase",letterSpacing:"0.5px",marginBottom:4}}>{title}</div>
              {note && <div style={{fontSize:11,color:"#888",marginBottom:10}}>{note}</div>}
              <table style={C.table}>
                <thead><tr>
                  <th style={C.th}>Company</th>
                  <th style={C.th}>District</th>
                  <th style={C.th}>Mandal</th>
                  <th style={C.th}>Inv</th>
                  <th style={C.th}>Emp</th>
                  <th style={C.th}>PCB</th>
                  <th style={C.th}>Estate</th>
                </tr></thead>
                <tbody>
                  {data.map(u=>(
                    <tr key={u[0]} style={{cursor:"pointer"}}>
                      <td style={C.td}>
                        <span style={{fontSize:12,fontWeight:500,color}} onClick={()=>{setSelected(u);setTab("search");}}>{u[1]}</span>
                        <div style={{fontSize:10,color:"#1D9E75",marginTop:2,cursor:"pointer"}} onClick={()=>openNetwork(u)}>Network</div>
                      </td>
                      <td style={{...C.td,fontSize:12}}>{districts[u[2]]||"--"}</td>
                      <td style={{...C.td,fontSize:11,color:"#888"}}>{mandals[u[3]]||"--"}</td>
                      <td style={{...C.td,fontSize:12,fontWeight:500}}>{fmt(u[7])}</td>
                      <td style={{...C.td,fontSize:12}}>{fmtE(u[8])}</td>
                      <td style={C.td}><span style={C.badge(PCB_COL[u[5]])}>{PCBS[u[5]]}</span></td>
                      <td style={{...C.td,fontSize:11}}>{u[10]?"Yes":"No"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );

          return (<>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
              <Section title={`MEGA companies (${mega.length})`} data={mega} color="#7F77DD" note="Anchor buyers - highest investment, set supply chain demand" />
              <Section title="HIGH PCB RISK - outside estates" data={redRisk} color="#E24B4A" note="RED tier pollution, not in industrial estate - supply chain risk" />
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
              <Section title="Top LARGE companies" data={large} color="#D85A30" note="Major buyers and potential consolidation targets" />
              <Section title="ODOP anchor companies" data={odopBig} color="#534AB7" note="One District One Product - priority supply chain nodes" />
            </div>
            <Section title="Women-led MEDIUM/LARGE/MEGA" data={womenLarge} color="#D85A30" note="Diversity-priority supplier pipeline" />
          </>);
        })()}

      </div>
    </div>
  );
}
