// frontend/src/pages/Departments/Fire/fireStationsMock.js
//
// Simulated Mumbai Fire Brigade operations for the Fire Stations page.
// Station names are real localities; coordinates, crew sizes and incident
// data are APPROXIMATE / SIMULATED and must not be used operationally.
//
// What makes it feel real:
//  - incidents are generated continuously (more in the evening peak) and move through
//    DISPATCHED -> EN_ROUTE -> ON_SCENE -> CONTAINED -> cleared
//  - larger "call levels" commit more engines, so stations actually run out of engines
//  - response times rise in the morning and evening traffic peaks
//  - state lives at module level, so it keeps running while you navigate between pages
//
// Same shapes as the /api/fire/stations/* endpoints, plus engines_available, zone and incident coordinates.

const TIME_SCALE = 5; // 1 real second = 5 simulated seconds, so changes are visible within minutes
const MIN = 60000;

// ---------- Random helpers (seeded for the initial picture, Math.random afterwards) ----------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
let rnd = mulberry32(20261006);
const between = (a, b) => a + rnd() * (b - a);
const pick = (arr) => arr[Math.floor(rnd() * arr.length)];
const pickWeighted = (items, weightOf) => {
  const total = items.reduce((s, i) => s + weightOf(i), 0);
  let r = rnd() * total;
  for (const i of items) { r -= weightOf(i); if (r <= 0) return i; }
  return items[items.length - 1];
};

// ---------- Stations ----------
const mk = (n, name, zone, latitude, longitude, engines, ladders, baseResp, risk, places, bias = {}) => ({
  entity_id: `stn${String(n).padStart(2, '0')}-fire-mumbai`,
  name: `${name} Fire Station`,
  short: name, zone, latitude, longitude, engines, ladders,
  crew: engines * 6 + 4,
  baseResp, risk, places, bias,
});

const STATIONS = [
  mk(1, 'Colaba', 'Island City', 18.9150, 72.8240, 3, 1, 6.2, 1.0, ['Colaba Causeway', 'Cuffe Parade Road', 'Sassoon Dock']),
  mk(2, 'Marine Lines', 'Island City', 18.9440, 72.8240, 3, 1, 6.0, 1.0, ['Princess Street', 'Charni Road Station Road', 'Dhobi Talao']),
  mk(3, 'Byculla', 'Island City', 18.9790, 72.8340, 5, 2, 5.8, 1.3, ['Dr. Ambedkar Road, Byculla', 'Mazgaon Dock Road', 'Nagpada Junction']),
  mk(4, 'Worli', 'Island City', 19.0100, 72.8180, 3, 1, 6.8, 1.0, ['Dr. Annie Besant Road', 'BDD Chawl, Worli', 'Senapati Bapat Marg, Lower Parel']),
  mk(5, 'Dadar', 'Island City', 19.0180, 72.8430, 4, 1, 7.0, 1.4, ['Dadar TT Circle', 'Ranade Road, Dadar West', 'Gokhale Road, Dadar']),
  mk(6, 'Wadala', 'Island City', 19.0170, 72.8590, 3, 1, 7.4, 1.0, ['Wadala Truck Terminal', 'Antop Hill', 'Bhakti Park']),
  mk(7, 'Mahim', 'Western Suburbs', 19.0410, 72.8400, 3, 1, 7.2, 1.5, ['Mahim Causeway', 'Dharavi Main Road', 'Kapad Bazar, Dharavi']),
  mk(8, 'Bandra', 'Western Suburbs', 19.0540, 72.8400, 3, 1, 7.6, 1.1, ['Linking Road, Bandra West', 'Waroda Road', 'Bandra Reclamation']),
  mk(9, 'Santacruz', 'Western Suburbs', 19.0800, 72.8420, 3, 1, 7.8, 1.0, ['Santacruz Station Road', 'Vakola Market', 'Kalina, CST Road']),
  mk(10, 'Andheri', 'Western Suburbs', 19.1130, 72.8420, 4, 2, 8.2, 1.5, ['Veera Desai Road', 'MIDC, Andheri East', 'Sakinaka Junction']),
  mk(11, 'Goregaon', 'Western Suburbs', 19.1630, 72.8490, 3, 1, 8.4, 1.1, ['Aarey Road', 'Link Road, Goregaon West', 'Film City Road']),
  mk(12, 'Malad', 'Western Suburbs', 19.1860, 72.8490, 4, 1, 8.8, 1.4, ['Malad Link Road', 'Marve Road', 'Mith Chowky']),
  mk(13, 'Borivali', 'Western Suburbs', 19.2300, 72.8580, 4, 2, 9.0, 1.2, ['SV Road, Borivali West', 'Gorai Road', 'National Park Gate, Borivali East']),
  mk(14, 'Kurla', 'Eastern Suburbs', 19.0700, 72.8800, 4, 1, 8.6, 1.5, ['LBS Marg, Kurla', 'Kurla Kamani Junction', 'Nehru Nagar']),
  mk(15, 'Ghatkopar', 'Eastern Suburbs', 19.0850, 72.9090, 3, 1, 8.4, 1.2, ['LBS Marg, Ghatkopar', 'Asalpha Village', 'Pant Nagar']),
  mk(16, 'Chembur', 'Eastern Suburbs', 19.0620, 72.8990, 3, 1, 8.0, 1.2, ['RC Marg, Chembur', 'Deonar Dumping Ground', 'Mahul Road'], { GARBAGE_FIRE: 4 }),
  mk(17, 'Powai', 'Eastern Suburbs', 19.1180, 72.9050, 2, 1, 8.8, 0.9, ['Hiranandani Gardens, Powai', 'JVLR, Powai', 'IIT Main Gate, Powai']),
  mk(18, 'Mulund', 'Eastern Suburbs', 19.1720, 72.9560, 3, 1, 9.2, 1.0, ['LBS Marg, Mulund West', 'Mulund Dumping Ground', 'Nahur Road'], { GARBAGE_FIRE: 4 }),
];
const byId = Object.fromEntries(STATIONS.map((s) => [s.entity_id, s]));

// ---------- Incident model ----------
const TYPES = {
  ELECTRICAL_FIRE: { w: 30, levels: [88, 11, 1], label: 'Electrical fire' },
  STRUCTURE_FIRE: { w: 14, levels: [45, 42, 13], label: 'Building fire' },
  GARBAGE_FIRE: { w: 12, levels: [95, 5, 0], label: 'Garbage fire' },
  VEHICLE_FIRE: { w: 10, levels: [92, 8, 0], label: 'Vehicle fire' },
  GAS_LEAK: { w: 8, levels: [85, 13, 2], label: 'Gas leak' },
  RESCUE: { w: 11, levels: [97, 3, 0], label: 'Rescue call' },
  BUILDING_COLLAPSE: { w: 2, levels: [10, 60, 30], label: 'Building collapse' },
};
const ENGINES_BY_LEVEL = { 1: 1, 2: 3, 3: 5 };
const ON_SCENE_MINUTES = { 1: [10, 32], 2: [35, 85], 3: [90, 170] };
const STATUS_ORDER = ['DISPATCHED', 'EN_ROUTE', 'ON_SCENE', 'CONTAINED'];

const hourNow = () => new Date().getHours();
const isPeak = (h) => (h >= 8 && h < 11) || (h >= 17 && h < 21);
const hourFactor = (h) => (h < 6 ? 0.4 : h < 10 ? 0.9 : h < 17 ? 1.0 : h < 23 ? 1.5 : 0.7);

let simNow = Date.now();
let lastReal = Date.now();
let seq = 12 + hourNow() * 3;
let evSeq = 0;
const active = [];
const events = [];
const todaySpawned = {};
const todayBase = {};
const hoursSoFar = new Date().getHours() + new Date().getMinutes() / 60;
STATIONS.forEach((s) => {
  todaySpawned[s.entity_id] = 0;
  todayBase[s.entity_id] = Math.round(s.risk * hoursSoFar * 0.2 + rnd() * 1.5);
});

const ymd = () => {
  const d = new Date();
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`;
};

function logEvent(kind, text, station_id, ts = Date.now()) {
  events.push({ id: ++evSeq, ts, kind, text, station_id });
  if (events.length > 60) events.shift();
}

function spawn(startIdx = 0, ts = Date.now()) {
  const countFor = (s) => active.filter((i) => i.station_id === s.entity_id).length;
  const st = pickWeighted(STATIONS, (s) => s.risk / (1 + 0.6 * countFor(s)));
  const type = pickWeighted(Object.keys(TYPES), (k) => TYPES[k].w * (st.bias[k] || 1));
  const [l1, l2] = TYPES[type].levels;
  const r = rnd() * 100;
  const level = r < l1 ? 1 : r < l1 + l2 ? 2 : 3;

  const durations = {
    DISPATCHED: between(1, 2) * MIN,
    EN_ROUTE: between(0.7, 1.3) * st.baseResp * MIN * (isPeak(hourNow()) ? 1.2 : 1),
    ON_SCENE: between(...ON_SCENE_MINUTES[level]) * MIN,
    CONTAINED: between(8, 20) * MIN,
  };
  const status = STATUS_ORDER[startIdx];
  const startedAt = startIdx === 0 ? simNow : simNow - rnd() * durations[status] * 0.7;
  const earlier = STATUS_ORDER.slice(0, startIdx).reduce((a, s) => a + durations[s], 0);
  const place = pick(st.places);

  active.push({
    incident_number: `FB-${ymd()}-${String(++seq).padStart(4, '0')}`,
    station_id: st.entity_id,
    incident_type: type,
    call_level: level,
    engines_assigned: ENGINES_BY_LEVEL[level],
    status,
    statusStartedAt: startedAt,
    reportedAt: startedAt - earlier,
    durations,
    place,
    zone: st.zone,
    latitude: st.latitude + between(-0.010, 0.010),
    longitude: st.longitude + between(-0.010, 0.010),
  });
  todaySpawned[st.entity_id] += 1;
  logEvent('NEW', `${TYPES[type].label} at ${place}. Level ${level} call, ${st.short} responding.`, st.entity_id, ts);
}

function progress() {
  for (let i = active.length - 1; i >= 0; i--) {
    const inc = active[i];
    const st = byId[inc.station_id];
    while (simNow >= inc.statusStartedAt + inc.durations[inc.status]) {
      const end = inc.statusStartedAt + inc.durations[inc.status];
      const idx = STATUS_ORDER.indexOf(inc.status);
      if (idx === STATUS_ORDER.length - 1) {
        logEvent('CLEARED', `${inc.place} cleared. ${st.short} back in service.`, st.entity_id);
        active.splice(i, 1);
        break;
      }
      inc.status = STATUS_ORDER[idx + 1];
      inc.statusStartedAt = end;
      if (inc.status === 'EN_ROUTE') logEvent('EN_ROUTE', `${st.short}: ${inc.engines_assigned} engine${inc.engines_assigned > 1 ? 's' : ''} en route to ${inc.place}.`, st.entity_id);
      if (inc.status === 'ON_SCENE') logEvent('ON_SCENE', `${st.short} crew on scene at ${inc.place}.`, st.entity_id);
      if (inc.status === 'CONTAINED') logEvent('CONTAINED', `Fire contained at ${inc.place}. Overhaul in progress.`, st.entity_id);
    }
  }
}

function advance() {
  const now = Date.now();
  const dt = Math.min(now - lastReal, 10 * MIN); // cap catch-up after long absences
  lastReal = now;
  let left = dt;
  while (left > 0) {
    const step = Math.min(left, 15000);
    left -= step;
    simNow += step * TIME_SCALE;
    progress();
    const mean = (step / 56000) * hourFactor(hourNow());
    if (active.length < 14 && rnd() < 1 - Math.exp(-mean)) spawn(0);
  }
}

// ---------- Initial picture: a handful of calls already in progress ----------
[0, 1, 1, 2, 2, 2, 3].forEach((startIdx, i, arr) => spawn(startIdx, Date.now() - (arr.length - i) * 75000));
rnd = Math.random;

// ---------- Views ----------
const noise = (s, bucket) => Math.sin(bucket * 0.7 + s.latitude * 100) * 0.45;

function stationView(s) {
  const mine = active.filter((i) => i.station_id === s.entity_id);
  const committed = Math.min(s.engines, mine.reduce((a, i) => a + (i.status === 'CONTAINED' ? 1 : i.engines_assigned), 0));
  const count = (st) => mine.filter((i) => i.status === st).length;
  const bucket = Math.floor(Date.now() / 300000);
  return {
    entity_id: s.entity_id,
    name: s.name,
    latitude: s.latitude,
    longitude: s.longitude,
    layer_group: 'fire',
    zone: s.zone,
    metadata: { engines: s.engines, ladders: s.ladders, crew: s.crew, zone: s.zone },
    active_incidents: mine.length,
    engines_available: s.engines - committed,
    avg_response_time: Number((s.baseResp + (isPeak(hourNow()) ? 1.6 : 0) + mine.length * 0.4 + noise(s, bucket)).toFixed(1)),
    today_incidents: todayBase[s.entity_id] + todaySpawned[s.entity_id],
    dispatched_incidents: count('DISPATCHED'),
    en_route_incidents: count('EN_ROUTE'),
    on_scene_incidents: count('ON_SCENE'),
    contained_incidents: count('CONTAINED'),
  };
}

const incidentView = (i) => ({
  incident_number: i.incident_number,
  station_id: i.station_id,
  station_name: byId[i.station_id].short,
  incident_type: i.incident_type,
  call_level: i.call_level,
  engines_assigned: i.engines_assigned,
  status: i.status,
  location: i.place,
  address: i.zone,
  latitude: i.latitude,
  longitude: i.longitude,
  elapsed_min: Math.max(0, Math.round((simNow - i.reportedAt) / MIN)),
});

// ---------- Public API (same method names the page uses for the live backend) ----------
export const fireSim = {
  async stations() { advance(); return STATIONS.map(stationView); },
  async summary() { return {}; },
  async resources() {
    advance();
    const live = active.filter((i) => ['EN_ROUTE', 'ON_SCENE'].includes(i.status));
    return {
      totalFireEngines: STATIONS.reduce((a, s) => a + s.engines, 0),
      fireTendersDispatched: live.reduce((a, i) => a + i.engines_assigned, 0),
      rescueUnitsDispatched: active.filter((i) => i.status !== 'CONTAINED' && (i.incident_type === 'RESCUE' || i.incident_type === 'BUILDING_COLLAPSE' || i.call_level >= 2)).length,
      waterTankersDispatched: live.filter((i) => i.call_level >= 2 || i.incident_type === 'GARBAGE_FIRE').length,
    };
  },
  async station(id) {
    advance();
    if (!byId[id]) throw new Error('Station not found');
    return stationView(byId[id]);
  },
  async stationIncidents(id) {
    advance();
    return active.filter((i) => i.station_id === id).map(incidentView).sort((a, b) => b.elapsed_min - a.elapsed_min);
  },
  async activeIncidents() { advance(); return active.map(incidentView); },
  async events(sinceId = 0) { return events.filter((e) => e.id > sinceId); },
};