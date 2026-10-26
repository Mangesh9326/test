import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Tooltip, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

/* ============================================================================
   FIRE OPERATIONS AGENT (simulation, local mock data only)

   How it works
   - Every scenario is a script: incidents with a report time, lifecycle timings,
     unit assignments and escalations. Nothing is stored as "current state".
   - The only live state is the simulation clock (t, in seconds). Incident status,
     unit counts, station load, KPIs, flags and the activity feed are all derived
     from (scenario, t). That makes speed control, scrubbing and replay free.
   - To add a scenario, push a new object into SCENARIOS (see the shape below).
   ============================================================================ */

// ---------------------------------------------------------------------------
// 1. HELPERS
// ---------------------------------------------------------------------------
const m = (min) => Math.round(min * 60); // minutes -> seconds
const U = (e = 0, r = 0, t = 0, a = 0, s = 0) => ({ e, r, t, a, s }); // unit bundle
const D = (dispatch, arrive, contain, close) => [m(dispatch), m(arrive), m(contain), m(close)]; // minutes after report

const UNIT_KEYS = ['e', 'r', 't', 'a', 's'];
const UNIT_LABEL = { e: 'Fire engines', r: 'Rescue vans', t: 'Water tankers', a: 'Aerial platforms', s: 'Specialist units' };
const UNIT_SHORT = { e: 'eng', r: 'rescue', t: 'tanker', a: 'aerial', s: 'specialist' };
const sum = (u) => UNIT_KEYS.reduce((a, k) => a + u[k], 0);

const pad = (n) => String(Math.floor(n)).padStart(2, '0');
const fmtClock = (s) => {
  const x = ((Math.floor(s) % 86400) + 86400) % 86400;
  return `${pad(x / 3600)}:${pad((x % 3600) / 60)}:${pad(x % 60)}`;
};
const fmtDur = (s) => {
  const x = Math.max(0, Math.floor(s));
  return `${pad(x / 60)}:${pad(x % 60)}`;
};
const fmtElapsed = (s) => {
  const x = Math.max(0, Math.floor(s));
  return `${Math.floor(x / 60)}m ${pad(x % 60)}s`;
};

const RESPONSE_TARGET = m(8);
const SPEEDS = [1, 5, 10, 30, 60];

const SEV = {
  CRITICAL: { label: 'Critical', rank: 0, p: 'P1', hex: '#ef4444', dot: 'bg-red-500', text: 'text-red-400', chip: 'text-red-300 bg-red-500/15 border-red-500/40' },
  HIGH: { label: 'High', rank: 1, p: 'P2', hex: '#f97316', dot: 'bg-orange-500', text: 'text-orange-400', chip: 'text-orange-300 bg-orange-500/15 border-orange-500/40' },
  MEDIUM: { label: 'Medium', rank: 2, p: 'P3', hex: '#eab308', dot: 'bg-yellow-400', text: 'text-yellow-400', chip: 'text-yellow-300 bg-yellow-500/10 border-yellow-500/30' },
  LOW: { label: 'Low', rank: 3, p: 'P4', hex: '#10b981', dot: 'bg-emerald-500', text: 'text-emerald-400', chip: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30' },
  INFO: { label: 'Info', rank: 4, p: '', hex: '#38bdf8', dot: 'bg-sky-400', text: 'text-sky-400', chip: 'text-sky-300 bg-sky-500/10 border-sky-500/30' },
};

const STATUS = {
  DISPATCHED: { label: 'Dispatched', dot: 'bg-sky-400' },
  EN_ROUTE: { label: 'En route', dot: 'bg-amber-400' },
  ON_SCENE: { label: 'On scene', dot: 'bg-red-400' },
  CONTAINED: { label: 'Contained', dot: 'bg-emerald-400' },
  CLOSED: { label: 'Closed', dot: 'bg-slate-500' },
};

const WORKLOAD = {
  NORMAL: { label: 'Normal', bar: 'bg-emerald-500', chip: 'text-emerald-300 bg-emerald-500/10', hex: '#10b981' },
  BUSY: { label: 'Busy', bar: 'bg-yellow-400', chip: 'text-yellow-300 bg-yellow-500/10', hex: '#eab308' },
  'HIGH LOAD': { label: 'High load', bar: 'bg-orange-500', chip: 'text-orange-300 bg-orange-500/15', hex: '#f97316' },
  CRITICAL: { label: 'Critical', bar: 'bg-red-500', chip: 'text-red-300 bg-red-500/15', hex: '#ef4444' },
};

// ---------------------------------------------------------------------------
// 2. MOCK DATA: STATIONS (fleet = engines, rescue, tankers, aerial, specialist)
// ---------------------------------------------------------------------------
const STATIONS = [
  { id: 'STN-01', name: 'Colaba', lat: 18.9067, lng: 72.8147, fleet: U(3, 1, 1, 0, 0) },
  { id: 'STN-02', name: 'Byculla', lat: 18.976, lng: 72.833, fleet: U(4, 1, 2, 1, 1) },
  { id: 'STN-03', name: 'Worli', lat: 19.0136, lng: 72.8174, fleet: U(3, 1, 1, 0, 0) },
  { id: 'STN-04', name: 'Dadar', lat: 19.0178, lng: 72.8478, fleet: U(3, 1, 1, 1, 0) },
  { id: 'STN-05', name: 'Bandra', lat: 19.0596, lng: 72.8295, fleet: U(3, 1, 1, 0, 0) },
  { id: 'STN-06', name: 'Sion', lat: 19.039, lng: 72.8619, fleet: U(2, 1, 1, 0, 0) },
  { id: 'STN-07', name: 'Kurla', lat: 19.0726, lng: 72.8845, fleet: U(3, 1, 1, 0, 1) },
  { id: 'STN-08', name: 'Chembur', lat: 19.0522, lng: 72.9005, fleet: U(3, 1, 1, 0, 1) },
  { id: 'STN-09', name: 'Andheri', lat: 19.1197, lng: 72.8464, fleet: U(4, 1, 2, 1, 1) },
  { id: 'STN-10', name: 'Ghatkopar', lat: 19.086, lng: 72.9081, fleet: U(2, 1, 1, 0, 0) },
  { id: 'STN-11', name: 'Goregaon', lat: 19.1663, lng: 72.8526, fleet: U(3, 1, 1, 0, 0) },
  { id: 'STN-12', name: 'Malad', lat: 19.186, lng: 72.8481, fleet: U(2, 1, 1, 0, 0) },
  { id: 'STN-13', name: 'Borivali', lat: 19.2307, lng: 72.8567, fleet: U(3, 1, 1, 1, 0) },
  { id: 'STN-14', name: 'Mulund', lat: 19.1726, lng: 72.9425, fleet: U(2, 1, 1, 0, 0) },
];
const STN = Object.fromEntries(STATIONS.map((s) => [s.id, s]));

// ---------------------------------------------------------------------------
// 3. MOCK DATA: SCENARIOS
//    incident: { id, cat, type, area, lat, lng, sev, at, d, assign, esc?, hazard?, notes }
//      at      seconds from scenario start (negative = already running at start)
//      d       [dispatch, arrive, contain, close] in seconds after report
//      assign  [[stationId, units], ...] (first entry is the primary station)
//      esc     [{ after, sev, add: [[stationId, units]], note }]
//    rec: { id, at, sev, title, reason, inc }
// ---------------------------------------------------------------------------
const SCENARIOS = [
  {
    id: 'routine',
    name: 'Routine shift',
    tag: 'Baseline',
    blurb: 'A normal weekday afternoon: a few working fires, routine rescues, and two stations running hot. Start here to learn the screen.',
    clock: 14 * 3600 + 15 * 60,
    duration: m(45),
    center: [19.076, 72.8777],
    zoom: 11,
    oos: {},
    incidents: [
      { id: 'FIR-2026-0142', cat: 'Fire', type: 'Electrical fire', area: 'Linking Rd, Bandra West', lat: 19.056, lng: 72.832, sev: 'MEDIUM', at: m(-34), d: D(1, 6, 22, 30), assign: [['STN-05', U(1)]], notes: 'Meter panel fire in a ground-floor shop. No injuries.' },
      { id: 'FIR-2026-0145', cat: 'Fire', type: 'Bus fire', area: 'Dr E Moses Rd, Worli', lat: 19.0169, lng: 72.8166, sev: 'HIGH', at: m(-27), d: D(1, 7, 24, 31), assign: [['STN-03', U(2, 1)]], hazard: { flag: 'Traffic impact', reason: 'Carriageway towards Haji Ali is blocked by the vehicle and hose lines.', action: 'Ask traffic control to divert via Annie Besant Rd until the road is clear.' }, notes: 'Passengers got out before the first unit arrived.' },
      { id: 'FIR-2026-0148', cat: 'Fire', type: 'Godown fire', area: 'Marol MIDC, Andheri East', lat: 19.1136, lng: 72.8697, sev: 'HIGH', at: m(-18.6), d: D(1, 9, 40, 52), assign: [['STN-09', U(3, 1, 2)], ['STN-11', U(1, 0, 1)]], esc: [{ after: m(22), sev: 'CRITICAL', add: [['STN-02', U(0, 0, 0, 1)]], note: 'Fire has spread to the adjoining godown. Aerial platform sent from Byculla.' }], hazard: { flag: 'Escalation risk', reason: 'Plastic and solvent stock inside. Dense smoke is drifting across Andheri-Kurla Rd.', action: 'Keep a second tanker on standby and confirm hydrant pressure with the water department.' }, notes: 'Plastic goods godown, around 4,000 sq ft. Two workers treated for smoke inhalation.' },
      { id: 'FIR-2026-0151', cat: 'Fire', type: 'Building fire', area: 'LBS Marg, Kurla West', lat: 19.07, lng: 72.8794, sev: 'HIGH', at: m(2), d: D(1.5, 8, 28, 40), assign: [['STN-07', U(2, 1)], ['STN-06', U(1)]], notes: 'Third floor of a seven-storey residential building. 14 residents evacuated.' },
      { id: 'FIR-2026-0153', cat: 'Fire', type: 'Warehouse fire', area: 'Chedda Nagar, Chembur', lat: 19.0525, lng: 72.9005, sev: 'HIGH', at: m(5), d: D(1, 11, 38, 50), assign: [['STN-08', U(2, 0, 1)], ['STN-10', U(1)]], notes: 'Cloth and packaging warehouse. Heavy smoke, no one inside.' },
      { id: 'FIR-2026-0156', cat: 'Fire', type: 'Cylinder fire', area: '90 Feet Rd, Dharavi', lat: 19.043, lng: 72.854, sev: 'MEDIUM', at: m(10), d: D(1, 6, 20, 32), assign: [['STN-06', U(1, 1, 1)]], esc: [{ after: m(12), sev: 'HIGH', add: [['STN-04', U(1, 0, 1)]], note: 'Fire is spreading across tin-roof units. Narrow lanes are limiting access.' }], hazard: { flag: 'Dense settlement', reason: 'Lanes are too narrow for engines. Hose lines must be laid over 150 m.', action: 'Stage a rescue van at the lane entrance and request a second tanker.' }, notes: 'Gas cylinder leak ignited in a ground-floor tenement.' },
      { id: 'FIR-2026-0158', cat: 'Fire', type: 'Short-circuit fire', area: 'Borivali East', lat: 19.23, lng: 72.865, sev: 'LOW', at: m(16), d: D(1, 9, 18, 28), assign: [['STN-13', U(1)]], notes: 'AC unit fire in a second-floor flat. Contained by residents before arrival.' },
      { id: 'FIR-2026-0160', cat: 'Fire', type: 'Garbage dump fire', area: 'Mulund dumping ground', lat: 19.165, lng: 72.954, sev: 'LOW', at: m(25), d: D(2, 14, 40, 70), assign: [['STN-14', U(1, 0, 1)]], notes: 'Surface fire with smouldering below. Expect a long overhaul.' },
      { id: 'RES-2026-0162', cat: 'Rescue', type: 'Tree fall on live wire', area: 'Hindu Colony, Dadar', lat: 19.024, lng: 72.844, sev: 'LOW', at: m(33), d: D(1, 6, 15, 22), assign: [['STN-04', U(0, 1)]], notes: 'Branch on a car and a sparking cable. Power utility notified.' },
      { id: 'RES-2026-0163', cat: 'Rescue', type: 'Lift rescue', area: 'Santacruz West', lat: 19.081, lng: 72.841, sev: 'LOW', at: m(38), d: D(1, 7, 12, 18), assign: [['STN-05', U(0, 1)]], notes: 'Two people stuck between floors. No medical issue.' },
    ],
    recs: [
      { id: 'R1', at: m(0.5), sev: 'LOW', title: 'Release Worli units', reason: 'Bus fire is contained and the road is clear. Worli can send back its two engines and the rescue van.', inc: 'FIR-2026-0145' },
      { id: 'R2', at: m(3.6), sev: 'CRITICAL', title: 'Review escalation at Marol MIDC', reason: 'Second godown is now involved after 22 minutes. Six units are committed and the Byculla aerial platform is on its way.', inc: 'FIR-2026-0148' },
      { id: 'R3', at: m(4), sev: 'HIGH', title: 'Andheri station is running low', reason: 'Only 3 of 9 units are free. Consider moving a Goregaon engine to cover Andheri West and Jogeshwari.', inc: 'FIR-2026-0148' },
      { id: 'R4', at: m(12.5), sev: 'HIGH', title: 'Dharavi fire is spreading', reason: 'Sion has no free units and the lanes are too narrow for engines. Dadar is sending an engine and tanker. Confirm hose-line access.', inc: 'FIR-2026-0156' },
    ],
  },

  {
    id: 'highrise',
    name: 'High-rise fire',
    tag: 'Escalating',
    blurb: 'A residential tower fire in Lower Parel climbs from Level 2 to Level 4. Watch aerial platforms and south Mumbai cover run thin.',
    clock: 21 * 3600 + 40 * 60,
    duration: m(50),
    center: [19.0, 72.835],
    zoom: 13,
    oos: {},
    incidents: [
      { id: 'FIR-2026-0311', cat: 'Fire', type: 'High-rise fire', area: 'Senapati Bapat Marg, Lower Parel', lat: 18.995, lng: 72.83, sev: 'HIGH', at: m(1), d: D(1.5, 7, 44, 70), assign: [['STN-03', U(2, 1, 1)], ['STN-02', U(1, 0, 0, 1)]], esc: [
        { after: m(6), sev: 'CRITICAL', add: [['STN-04', U(2, 1, 1, 1)], ['STN-01', U(1, 0, 1)]], note: 'Level 3 declared. Fire on the 22nd floor with smoke up to the 27th. Residents on terraces.' },
        { after: m(15), sev: 'CRITICAL', add: [['STN-05', U(2, 1, 1)], ['STN-09', U(1, 0, 0, 1, 1)], ['STN-02', U(0, 0, 1)]], note: 'Level 4 (Brigade call). More breathing-apparatus sets and a second platform ordered.' },
      ], hazard: { flag: 'High-rise protocol', reason: 'Fire is above the reach of standard ladders. Stairwell smoke is blocking evacuation, so aerial platforms are the only external option.', action: 'Confirm refuge-floor headcount with the building warden and stage rescue vans at the podium.' }, notes: 'Caller on the 23rd floor. Fire alarm and sprinklers reported not working.' },
      { id: 'FIR-2026-0313', cat: 'Fire', type: 'Electrical fire', area: 'Prabhadevi', lat: 19.015, lng: 72.83, sev: 'MEDIUM', at: m(9), d: D(1, 8, 25, 35), assign: [['STN-03', U(1)]], notes: 'Transformer fire near a commercial block. Power cut to the lane.' },
      { id: 'RES-2026-0314', cat: 'Rescue', type: 'Lift rescue', area: 'Mahim', lat: 19.041, lng: 72.84, sev: 'LOW', at: m(12), d: D(1, 7, 14, 22), assign: [['STN-06', U(0, 1)]], notes: 'Three people stuck in a lift. Caller is calm.' },
      { id: 'FIR-2026-0315', cat: 'Fire', type: 'Auto-rickshaw fire', area: 'Colaba Causeway', lat: 18.922, lng: 72.833, sev: 'LOW', at: m(20), d: D(1, 6, 12, 20), assign: [['STN-01', U(1)]], notes: 'CNG auto fire. Driver unhurt.' },
      { id: 'FIR-2026-0316', cat: 'Fire', type: 'Basement smoke', area: 'Mahalaxmi', lat: 18.982, lng: 72.824, sev: 'MEDIUM', at: m(28), d: D(1, 7, 20, 30), assign: [['STN-02', U(1)]], notes: 'Smoke from a parking basement. Source not yet found.' },
    ],
    recs: [
      { id: 'R1', at: m(6.5), sev: 'CRITICAL', title: 'Stage a second aerial platform', reason: 'Fire is above the 20th floor. Byculla and Dadar platforms are committed; Andheri and Borivali hold the remaining two.', inc: 'FIR-2026-0311' },
      { id: 'R2', at: m(10), sev: 'HIGH', title: 'Worli station is depleted', reason: 'All Worli units are out. Request cover from Sion for the Prabhadevi transformer fire.', inc: 'FIR-2026-0313' },
      { id: 'R3', at: m(15.5), sev: 'CRITICAL', title: 'South Mumbai cover is thin', reason: 'Colaba, Worli and Dadar have little left. Move Sion and Kurla engines to cover the Byculla-Dadar belt.', inc: 'FIR-2026-0311' },
      { id: 'R4', at: m(18), sev: 'HIGH', title: 'Coordinate a traffic diversion', reason: 'Apparatus is blocking one lane of Senapati Bapat Marg. Alert traffic police for a one-lane diversion.', inc: 'FIR-2026-0311' },
    ],
  },

  {
    id: 'monsoon',
    name: 'Monsoon surge',
    tag: 'Surge',
    blurb: 'Red-alert rain. Collapses, flooded subways and tree falls arrive together, response times stretch, and two stations lose an engine to waterlogging.',
    clock: 17 * 3600 + 30 * 60,
    duration: m(60),
    center: [19.06, 72.87],
    zoom: 11,
    oos: { 'STN-06': { e: 1 }, 'STN-07': { e: 1 } },
    incidents: [
      { id: 'RES-2026-0401', cat: 'Rescue', type: 'Tree fall blocking road', area: 'Matunga', lat: 19.027, lng: 72.857, sev: 'MEDIUM', at: m(-12), d: D(2, 14, 30, 38), assign: [['STN-04', U(0, 1)]], notes: 'Large rain tree across a two-way road. Bus stuck behind it.' },
      { id: 'RES-2026-0403', cat: 'Rescue', type: 'Stranded vehicles', area: 'Hindmata, Dadar East', lat: 19.0105, lng: 72.8445, sev: 'HIGH', at: m(-5), d: D(2, 18, 45, 60), assign: [['STN-04', U(1, 0, 1)], ['STN-06', U(0, 1)]], hazard: { flag: 'Flooded access', reason: 'Hindmata is under 1.2 m of water. Units must come via Parel, adding 8 to 10 minutes.', action: 'Redirect arriving units via Dr Ambedkar Rd and request a dewatering pump.' }, notes: 'Cars and a school bus stranded. Children on board.' },
      { id: 'RES-2026-0405', cat: 'Rescue', type: 'Retaining wall collapse', area: 'Kannamwar Nagar, Vikhroli', lat: 19.11, lng: 72.928, sev: 'CRITICAL', at: m(4), d: D(2, 16, 95, 110), assign: [['STN-10', U(1, 1)], ['STN-08', U(1, 1, 1)]], esc: [{ after: m(10), sev: 'CRITICAL', add: [['STN-14', U(1, 1)]], note: 'Three houses buried. NDRF team requested.' }], hazard: { flag: 'Mass casualty risk', reason: 'Several people are trapped under debris. More slope movement is likely in continued rain.', action: 'Alert the nearest trauma hospital and confirm the NDRF arrival time.' }, notes: 'Hillside slum. Residents are digging by hand.' },
      { id: 'RES-2026-0406', cat: 'Rescue', type: 'Building collapse', area: 'Bhendi Bazaar', lat: 18.963, lng: 72.83, sev: 'CRITICAL', at: m(12), d: D(2, 12, 150, 170), assign: [['STN-02', U(2, 1, 1, 0, 1)], ['STN-01', U(1, 1)]], esc: [{ after: m(8), sev: 'CRITICAL', add: [['STN-03', U(1, 0, 1)]], note: 'Over 20 residents believed inside. Search dogs requested.' }], hazard: { flag: 'Mass casualty risk', reason: 'Four-storey old building with heavy debris. Narrow approach lanes slow heavy rescue equipment.', action: 'Stage ambulances on Mohammed Ali Rd and request the heavy rescue crane.' }, notes: 'Cessed building, collapsed after nine hours of rain.' },
      { id: 'FIR-2026-0407', cat: 'Fire', type: 'Short circuit in chawl', area: 'Kurla West', lat: 19.07, lng: 72.879, sev: 'MEDIUM', at: m(15), d: D(2, 19, 35, 50), assign: [['STN-07', U(1)]], notes: 'Waterlogged wiring sparked. Flooded lane delayed the first unit.' },
      { id: 'RES-2026-0409', cat: 'Rescue', type: 'Tree fall', area: 'Reclamation, Bandra', lat: 19.05, lng: 72.828, sev: 'LOW', at: m(18), d: D(2, 11, 22, 30), assign: [['STN-05', U(0, 1)]], notes: 'Tree on a parked car. No injuries.' },
      { id: 'RES-2026-0411', cat: 'Rescue', type: 'Persons trapped in car', area: 'Andheri Subway', lat: 19.1196, lng: 72.8497, sev: 'HIGH', at: m(20), d: D(2, 15, 40, 55), assign: [['STN-09', U(1, 1, 1)]], hazard: { flag: 'Flooded access', reason: 'Subway is closed and under water. Pumps are needed before rescuers can reach the vehicles.', action: 'Send the tanker with the high-capacity pump and keep a boat team ready.' }, notes: 'Two cars submerged. Occupants on the roof of one.' },
      { id: 'RES-2026-0413', cat: 'Rescue', type: 'Lift stuck in power cut', area: 'Goregaon West', lat: 19.165, lng: 72.84, sev: 'LOW', at: m(27), d: D(2, 12, 20, 28), assign: [['STN-11', U(0, 1)]], notes: 'Elderly resident stuck. Medical check requested.' },
      { id: 'RES-2026-0415', cat: 'Rescue', type: 'Building tilt', area: 'Malvani, Malad West', lat: 19.199, lng: 72.818, sev: 'MEDIUM', at: m(31), d: D(2, 14, 50, 65), assign: [['STN-12', U(1, 1)]], esc: [{ after: m(9), sev: 'HIGH', add: [], note: 'Cracks are widening. 40 residents evacuated.' }], notes: 'Three-storey structure leaning after foundation washout.' },
      { id: 'FIR-2026-0417', cat: 'Fire', type: 'Substation fire', area: 'Sion', lat: 19.039, lng: 72.862, sev: 'HIGH', at: m(38), d: D(2, 17, 44, 58), assign: [['STN-06', U(1, 0, 1)], ['STN-04', U(1)]], notes: 'Flooded substation arcing. Power utility on the way.' },
      { id: 'RES-2026-0419', cat: 'Rescue', type: 'Road cave-in', area: 'Worli Naka', lat: 19.0, lng: 72.818, sev: 'MEDIUM', at: m(45), d: D(2, 10, 30, 40), assign: [['STN-03', U(1, 1)]], notes: 'Section of road gave way. Bus partly into the pit.' },
    ],
    recs: [
      { id: 'R1', at: m(6), sev: 'HIGH', title: 'Avoid the Hindmata approach', reason: 'The underpass is flooded. Route Dadar and Sion units via Parel and Dr Ambedkar Rd.', inc: 'RES-2026-0403' },
      { id: 'R2', at: m(12.5), sev: 'CRITICAL', title: 'Heavy rescue is split across two sites', reason: 'Vikhroli and Bhendi Bazaar both need specialist crews. Request mutual aid from Thane for Vikhroli.', inc: 'RES-2026-0406' },
      { id: 'R3', at: m(22), sev: 'HIGH', title: 'Response times are above target', reason: 'Flooded arterials are pushing arrival times well past 08:00. Pre-position engines at high-ground posts in Sion and Kurla.', inc: null },
      { id: 'R4', at: m(40), sev: 'MEDIUM', title: 'Two engines are out of service', reason: 'Sion and Kurla bays are waterlogged. Keep crews at forward posts until the water recedes.', inc: null },
    ],
  },

  {
    id: 'chemical',
    name: 'Chemical fire',
    tag: 'Hazmat',
    blurb: 'A solvent unit burns at night in the Mahul industrial belt. Specialist units run out and a toxic plume drifts towards residential areas.',
    clock: 2 * 3600 + 10 * 60,
    duration: m(55),
    center: [19.025, 72.89],
    zoom: 12,
    oos: {},
    incidents: [
      { id: 'FIR-2026-0501', cat: 'Hazmat', type: 'Chemical storage fire', area: 'Mahul industrial belt', lat: 19.001, lng: 72.89, sev: 'CRITICAL', at: m(2), d: D(2, 10, 60, 80), assign: [['STN-08', U(2, 1, 1, 0, 1)], ['STN-07', U(1, 0, 1, 0, 1)]], esc: [
        { after: m(5), sev: 'CRITICAL', add: [['STN-02', U(1, 0, 0, 0, 1)], ['STN-06', U(1, 0, 1)]], note: 'Level 4 hazmat call. Drum explosions heard. Shelter-in-place advisory for a 500 m radius.' },
        { after: m(18), sev: 'CRITICAL', add: [['STN-09', U(1, 0, 0, 0, 1)], ['STN-10', U(1, 0, 1)]], note: 'Foam stocks running low. Industrial foam requested from refinery partners.' },
      ], hazard: { flag: 'Hazmat protocol', reason: 'Toxic smoke plume is drifting east towards Chembur and Mankhurd. Specialist units and breathing apparatus are required.', action: 'Issue a shelter-in-place alert and notify the pollution control board and nearby hospitals.' }, notes: 'Solvent drums on site. Night shift of 12 workers, 9 accounted for.' },
      { id: 'HAZ-2026-0502', cat: 'Hazmat', type: 'Gas leak report', area: 'Mahul village', lat: 19.008, lng: 72.896, sev: 'HIGH', at: m(9), d: D(2, 9, 30, 42), assign: [['STN-10', U(1, 1)]], notes: 'Residents report a sharp smell and watering eyes.' },
      { id: 'FIR-2026-0503', cat: 'Fire', type: 'Vehicle fire', area: 'Eastern Express Hwy, Sion', lat: 19.045, lng: 72.87, sev: 'MEDIUM', at: m(15), d: D(1, 7, 18, 28), assign: [['STN-06', U(1)]], notes: 'Truck fire on the highway shoulder. Lane closed.' },
      { id: 'FIR-2026-0504', cat: 'Fire', type: 'Smoke complaint', area: 'Wadala', lat: 19.017, lng: 72.86, sev: 'LOW', at: m(25), d: D(1, 8, 15, 24), assign: [['STN-04', U(1)]], notes: 'Residents report smoke. Possibly drifting from Mahul.' },
      { id: 'FIR-2026-0505', cat: 'Fire', type: 'Shop fire', area: 'Kurla West', lat: 19.07, lng: 72.879, sev: 'MEDIUM', at: m(33), d: D(1, 7, 22, 32), assign: [['STN-07', U(1)]], notes: 'Shutter-down shop fire. Adjacent units evacuated.' },
    ],
    recs: [
      { id: 'R1', at: m(7.5), sev: 'CRITICAL', title: 'Issue a shelter-in-place alert', reason: 'Plume is drifting towards Chembur and Mankhurd. Alert ward officers and nearby hospitals.', inc: 'FIR-2026-0501' },
      { id: 'R2', at: m(10), sev: 'HIGH', title: 'Gas leak near the fire scene', reason: 'Residents report fumes in Mahul village. Keep the crew upwind and monitor gas levels.', inc: 'HAZ-2026-0502' },
      { id: 'R3', at: m(20.5), sev: 'CRITICAL', title: 'Specialist units are exhausted', reason: 'All four specialist units are committed. Request foam tenders and hazmat teams from neighbouring corporations.', inc: 'FIR-2026-0501' },
      { id: 'R4', at: m(30), sev: 'MEDIUM', title: 'Plan crew rotation', reason: 'First crews have worked 28 minutes in breathing apparatus. Rotate before cylinders run low.', inc: 'FIR-2026-0501' },
    ],
  },

  {
    id: 'visarjan',
    name: 'Visarjan crowd',
    tag: 'Crowd',
    blurb: 'Ganesh immersion day. Engines are pre-posted at the beaches while crowd incidents, a pandal fire and a boat capsize test cover and coordination.',
    clock: 15 * 3600,
    duration: m(60),
    center: [19.02, 72.83],
    zoom: 12,
    oos: {},
    incidents: [
      { id: 'STB-2026-0601', cat: 'Standby', type: 'Standby post', area: 'Girgaon Chowpatty', lat: 18.954, lng: 72.813, sev: 'LOW', at: m(-120), d: D(1, 5, 590, 600), assign: [['STN-01', U(1, 0, 1)]], notes: 'Immersion point post with a tanker and one engine.' },
      { id: 'STB-2026-0602', cat: 'Standby', type: 'Standby post', area: 'Dadar Chowpatty', lat: 19.021, lng: 72.833, sev: 'LOW', at: m(-120), d: D(1, 5, 590, 600), assign: [['STN-04', U(1)]], notes: 'Immersion point post with one engine.' },
      { id: 'STB-2026-0603', cat: 'Standby', type: 'Standby post', area: 'Juhu Chowpatty', lat: 19.099, lng: 72.826, sev: 'LOW', at: m(-120), d: D(1, 5, 590, 600), assign: [['STN-09', U(1)]], notes: 'Immersion point post with one engine.' },
      { id: 'STB-2026-0604', cat: 'Standby', type: 'Standby post', area: 'Versova', lat: 19.131, lng: 72.814, sev: 'LOW', at: m(-120), d: D(1, 5, 590, 600), assign: [['STN-11', U(1, 0, 1)]], notes: 'Immersion point post with an engine and a tanker.' },
      { id: 'FIR-2026-0610', cat: 'Fire', type: 'Pandal short circuit', area: 'Lalbaug', lat: 18.987, lng: 72.834, sev: 'MEDIUM', at: m(6), d: D(1, 6, 16, 26), assign: [['STN-02', U(1, 1)]], hazard: { flag: 'Crowd density', reason: 'Thousands of devotees are in the lane. Engines cannot enter until police open a path.', action: 'Ask police to clear a path and keep hose lines ready from the pandal side.' }, notes: 'Wiring fire at the back of a decorated pandal. Idol area is clear.' },
      { id: 'RES-2026-0611', cat: 'Rescue', type: 'Person drowning', area: 'Girgaon Chowpatty', lat: 18.954, lng: 72.813, sev: 'HIGH', at: m(14), d: D(1, 4, 12, 20), assign: [['STN-01', U(0, 1)]], notes: 'Devotee swept out during immersion. Lifeguards searching.' },
      { id: 'RES-2026-0612', cat: 'Rescue', type: 'Crowd crush', area: 'Dadar Chowpatty approach', lat: 19.0235, lng: 72.8355, sev: 'HIGH', at: m(24), d: D(1, 6, 30, 45), assign: [['STN-04', U(0, 1)], ['STN-06', U(1, 1)]], esc: [{ after: m(8), sev: 'CRITICAL', add: [['STN-05', U(0, 1)]], note: 'Surge at the barricade. Ambulances requested and police crowd control deployed.' }], hazard: { flag: 'Mass casualty risk', reason: 'Crowd is compressed at the barricade near the immersion ramp and people have fallen.', action: 'Stage ambulances at both ends of the approach and ask police to stop inflow.' }, notes: 'Reported by a police constable on the ramp.' },
      { id: 'RES-2026-0613', cat: 'Rescue', type: 'Boat capsize', area: 'Juhu Chowpatty', lat: 19.0995, lng: 72.8265, sev: 'HIGH', at: m(33), d: D(1, 5, 25, 40), assign: [['STN-09', U(0, 1)]], notes: 'Small boat overturned near the immersion point. Six people in the water.' },
      { id: 'FIR-2026-0614', cat: 'Fire', type: 'Tempo fire on route', area: 'Parel', lat: 19.0, lng: 72.84, sev: 'MEDIUM', at: m(42), d: D(1, 7, 18, 28), assign: [['STN-03', U(1)]], notes: 'Procession tempo caught fire. Idol removed safely.' },
    ],
    recs: [
      { id: 'R1', at: m(6.5), sev: 'HIGH', title: 'Crowd is blocking access at Lalbaug', reason: 'Engine cannot reach the pandal. Ask police for a cleared lane before committing more units.', inc: 'FIR-2026-0610' },
      { id: 'R2', at: m(24.5), sev: 'CRITICAL', title: 'Move a rescue van to Dadar Chowpatty', reason: 'Crowd crush on the approach. Bandra can send a rescue van; Dadar has no spare vans.', inc: 'RES-2026-0612' },
      { id: 'R3', at: m(35), sev: 'MEDIUM', title: 'Rotate standby crews', reason: 'Juhu and Girgaon crews have been on post for more than three hours. Plan relief before the evening peak.', inc: null },
    ],
  },
];

// ---------------------------------------------------------------------------
// 4. SIMULATION ENGINE (pure functions: state = f(scenario, t))
// ---------------------------------------------------------------------------
function computeState(sc, t) {
  const sMap = {};
  STATIONS.forEach((s) => {
    const oos = sc.oos?.[s.id] || {};
    const total = U();
    UNIT_KEYS.forEach((k) => { total[k] = Math.max(0, s.fleet[k] - (oos[k] || 0)); });
    sMap[s.id] = { ...s, total, oos, deployed: U(), ids: new Set() };
  });

  const incidents = [];
  for (const i of sc.incidents) {
    const el = t - i.at;
    if (el < 0) continue;
    const [dispatch, arrive, contain, close] = i.d;
    const status = el >= close ? 'CLOSED' : el >= contain ? 'CONTAINED' : el >= arrive ? 'ON_SCENE' : el >= dispatch ? 'EN_ROUTE' : 'DISPATCHED';

    const dep = {};
    const addDep = (list) => list.forEach(([sid, u]) => {
      dep[sid] = dep[sid] || U();
      UNIT_KEYS.forEach((k) => { dep[sid][k] += u[k]; });
    });
    addDep(i.assign);
    let sev = i.sev;
    const applied = (i.esc || []).filter((x) => el >= x.after);
    applied.forEach((x) => { sev = x.sev; addDep(x.add); });

    const units = U();
    Object.entries(dep).forEach(([sid, u]) => {
      UNIT_KEYS.forEach((k) => { units[k] += u[k]; });
      if (status !== 'CLOSED') {
        UNIT_KEYS.forEach((k) => { sMap[sid].deployed[k] += u[k]; });
        sMap[sid].ids.add(i.id);
      }
    });

    incidents.push({
      ...i, el, status, sev, dep, units, applied,
      unitTotal: sum(units),
      priority: SEV[sev].p,
      primary: i.assign[0][0],
      reportedAbs: sc.clock + i.at,
      eta: status === 'DISPATCHED' || status === 'EN_ROUTE' ? arrive - el : null,
    });
  }

  const stations = Object.values(sMap).map((s) => {
    const totalN = sum(s.total);
    const depN = Math.min(sum(s.deployed), totalN);
    const free = U();
    UNIT_KEYS.forEach((k) => { free[k] = Math.max(0, s.total[k] - s.deployed[k]); });
    const freeN = sum(free);
    const util = totalN ? depN / totalN : 0;
    const workload = util < 0.35 ? 'NORMAL' : util < 0.6 ? 'BUSY' : util < 0.85 ? 'HIGH LOAD' : 'CRITICAL';
    const status = freeN === 0 ? 'DEPLETED' : util >= 0.6 ? 'LIMITED' : 'OPERATIONAL';
    return { ...s, active: s.ids.size, totalN, depN, free, freeN, util, workload, status };
  });
  const stnById = Object.fromEntries(stations.map((s) => [s.id, s]));

  const fleet = {};
  UNIT_KEYS.forEach((k) => {
    let nominal = 0, oos = 0, dep = 0;
    stations.forEach((s) => {
      nominal += STN[s.id].fleet[k];
      oos += s.oos[k] || 0;
      dep += Math.min(s.deployed[k], s.total[k]);
    });
    fleet[k] = { nominal, oos, dep, avail: nominal - oos - dep };
  });

  const act = incidents.filter((i) => i.status !== 'CLOSED' && i.cat !== 'Standby');
  const real = incidents.filter((i) => i.cat !== 'Standby');
  const arrived = real.filter((i) => i.el >= i.d[1]);
  const avgResp = arrived.length ? arrived.reduce((a, i) => a + i.d[1], 0) / arrived.length : null;
  const sceneUnits = incidents.filter((i) => i.status === 'ON_SCENE' || i.status === 'CONTAINED').reduce((a, i) => a + i.unitTotal, 0);
  const routeUnits = incidents.filter((i) => i.status === 'EN_ROUTE' || i.status === 'DISPATCHED').reduce((a, i) => a + i.unitTotal, 0);
  const inService = UNIT_KEYS.reduce((a, k) => a + fleet[k].nominal - fleet[k].oos, 0);

  const kpis = {
    active: act.length,
    newLast10: real.filter((i) => i.el <= m(10)).length,
    critical: act.filter((i) => i.sev === 'CRITICAL').length,
    criticalOnScene: act.filter((i) => i.sev === 'CRITICAL' && i.status === 'ON_SCENE').length,
    high: act.filter((i) => i.sev === 'HIGH').length,
    deployed: stations.reduce((a, s) => a + s.depN, 0),
    inService,
    onScene: sceneUnits,
    enRoute: routeUnits,
    underLoad: stations.filter((s) => s.workload === 'HIGH LOAD' || s.workload === 'CRITICAL').length,
    avgResp,
  };

  return { incidents, stations, stnById, fleet, kpis, act };
}

function deriveFlag(i, stnById, aid) {
  if (i.status === 'CLOSED') return null;
  if (i.status === 'CONTAINED') {
    return { text: 'Ready to stand down', level: 'LOW', reason: 'Fire is contained. Units can be released once overhaul is complete.', action: 'Confirm with the incident commander and release surplus units to their stations.' };
  }
  if (i.hazard) return { text: i.hazard.flag, level: i.sev, reason: i.hazard.reason, action: i.hazard.action };
  if (i.cat === 'Standby') return null;
  const stn = stnById[i.primary];
  if (i.sev === 'CRITICAL' && i.status === 'ON_SCENE' && i.el > m(15)) {
    return { text: 'Escalation risk', level: 'CRITICAL', reason: 'Critical incident on scene for over 15 minutes with several units committed.', action: 'Review the incident commander plan and pre-alert the next-nearest station.' };
  }
  if (i.eta != null && i.eta > m(10) && (i.sev === 'HIGH' || i.sev === 'CRITICAL')) {
    return { text: 'Slow response', level: 'HIGH', reason: `First unit is still ${fmtDur(i.eta)} away.`, action: 'Check for an alternate route or send the nearest free unit.' };
  }
  if (stn && (stn.workload === 'HIGH LOAD' || stn.workload === 'CRITICAL')) {
    const asked = aid.has(i.id);
    return { text: asked ? 'Mutual aid requested' : 'Station stretched', level: 'HIGH', reason: `${stn.name} has ${stn.freeN} of ${stn.totalN} units free.`, action: asked ? 'Waiting for a neighbouring corporation to confirm units.' : 'Consider cover moves or mutual aid for this district.' };
  }
  if (i.unitTotal >= 8) {
    return { text: 'High resource demand', level: 'HIGH', reason: `${i.unitTotal} units are committed to a single incident.`, action: 'Check that remaining stations can still cover their own districts.' };
  }
  return null;
}

const unitText = (assign) => {
  const n = assign.reduce((a, [, u]) => a + sum(u), 0);
  const names = [...new Set(assign.map(([s]) => STN[s].name))];
  return `${n} unit${n === 1 ? '' : 's'} from ${names.join(', ')}`;
};

function buildEvents(sc) {
  const ev = [];
  sc.incidents.forEach((i) => {
    const [dispatch, arrive, contain, close] = i.d;
    const base = { id: i.id };
    if (i.cat === 'Standby') {
      ev.push({ ...base, time: i.at + arrive, type: 'STANDBY POST', sev: 'INFO', desc: `${i.area}: ${unitText(i.assign)} in position.` });
      return;
    }
    ev.push({ ...base, time: i.at, type: 'NEW INCIDENT', sev: i.sev, desc: `${i.type} at ${i.area}.` });
    ev.push({ ...base, time: i.at + dispatch, type: 'DISPATCHED', sev: 'INFO', desc: `${unitText(i.assign)} sent to ${i.id}.` });
    ev.push({ ...base, time: i.at + arrive, type: 'ON SCENE', sev: 'INFO', desc: `First unit reached ${i.id} after ${fmtDur(arrive)}.` });
    (i.esc || []).forEach((x) => ev.push({ ...base, time: i.at + x.after, type: 'ESCALATION', sev: x.sev, desc: `${i.id}: ${x.note}` }));
    ev.push({ ...base, time: i.at + contain, type: 'CONTAINED', sev: 'LOW', desc: `${i.id} is contained.` });
    ev.push({ ...base, time: i.at + close, type: 'CLOSED', sev: 'INFO', desc: `${i.id} closed. Units back at station.` });
  });
  sc.recs.forEach((r) => ev.push({ id: r.inc, time: r.at, type: 'AGENT FLAG', sev: r.sev, desc: r.title }));
  return ev;
}

const byPriority = (a, b) => SEV[a.sev].rank - SEV[b.sev].rank || b.el - a.el;

function buildInsights(state) {
  const { stations, kpis, fleet, act } = state;
  const out = [];
  const hot = stations.filter((s) => s.workload === 'HIGH LOAD' || s.workload === 'CRITICAL');
  out.push(hot.length ? `${hot.length} of ${stations.length} stations are on high load or critical: ${hot.map((s) => s.name).join(', ')}.` : `All ${stations.length} stations have spare capacity.`);
  if (act.length) out.push(`${kpis.critical} of ${act.length} active incidents are critical (${Math.round((kpis.critical / act.length) * 100)}%).`);
  const top = [...stations].sort((a, b) => b.depN - a.depN)[0];
  if (top && top.depN > 0) out.push(`${top.name} has the most units out: ${top.depN} of ${top.totalN}.`);
  if (fleet.a.nominal - fleet.a.oos > 0 && fleet.a.avail <= 1 && fleet.a.dep > 0) out.push(`Only ${fleet.a.avail} of ${fleet.a.nominal - fleet.a.oos} aerial platforms are free.`);
  if (fleet.s.nominal > 0 && fleet.s.avail === 0 && fleet.s.dep > 0) out.push('No specialist units are free. A new hazmat or heavy-rescue call would need mutual aid.');
  if (kpis.avgResp != null && kpis.avgResp > RESPONSE_TARGET) out.push(`Average response is ${fmtDur(kpis.avgResp - RESPONSE_TARGET)} over the ${fmtDur(RESPONSE_TARGET)} target.`);
  const longest = [...act].sort((a, b) => b.el - a.el)[0];
  if (longest) out.push(`${longest.id} has been open the longest: ${fmtElapsed(longest.el)}.`);
  return out.slice(0, 5);
}

// ---------------------------------------------------------------------------
// 5. MAP HELPERS
// ---------------------------------------------------------------------------
const iconCache = new Map();
function incidentIcon(sev, status, selected) {
  const key = `${sev}|${status}|${selected}`;
  if (iconCache.has(key)) return iconCache.get(key);
  const done = status === 'CONTAINED' || status === 'CLOSED';
  const c = done ? '#64748b' : SEV[sev].hex;
  const size = sev === 'CRITICAL' ? 20 : sev === 'HIGH' ? 16 : 13;
  const responding = status === 'DISPATCHED' || status === 'EN_ROUTE';
  const pulse = sev === 'CRITICAL' && !done ? 'agt-pulse' : '';
  const ring = selected ? '0 0 0 3px rgba(255,255,255,.4),' : '';
  const html = `<div class="agt-dot ${pulse}" style="--c:${c};width:${size}px;height:${size}px;background:${responding ? '#0f172a' : c};border:2px ${responding ? 'dashed' : 'solid'} ${selected ? '#fff' : c};box-shadow:${ring}0 0 10px ${done ? 'transparent' : c}"></div>`;
  const icon = L.divIcon({ className: 'agt-marker', html, iconSize: [size, size], iconAnchor: [size / 2, size / 2] });
  iconCache.set(key, icon);
  return icon;
}
const stationIconCache = new Map();
function stationIcon(workload) {
  if (stationIconCache.has(workload)) return stationIconCache.get(workload);
  const c = WORKLOAD[workload].hex;
  const icon = L.divIcon({
    className: 'agt-marker',
    html: `<div style="width:11px;height:11px;border-radius:2px;background:${c};border:1.5px solid #0f172a;box-shadow:0 0 0 1px ${c}"></div>`,
    iconSize: [11, 11], iconAnchor: [5.5, 5.5],
  });
  stationIconCache.set(workload, icon);
  return icon;
}

function MapController({ view, focus }) {
  const map = useMap();
  useEffect(() => { map.setView(view.center, view.zoom, { animate: true }); }, [view.key]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (focus) map.flyTo([focus.lat, focus.lng], Math.max(map.getZoom(), focus.zoom || 13), { duration: 0.7 });
  }, [focus?.k]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

// ---------------------------------------------------------------------------
// 6. SMALL UI PIECES
// ---------------------------------------------------------------------------
const focusRing = 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500/70';

const Chip = ({ tone = 'slate', pulse, children }) => {
  const tones = {
    slate: 'text-slate-300 bg-slate-800 border-slate-700',
    emerald: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30',
    amber: 'text-amber-300 bg-amber-500/10 border-amber-500/30',
  };
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}>
      {pulse && <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse motion-reduce:animate-none" />}
      {children}
    </span>
  );
};

const SevBadge = ({ sev }) => (
  <span className={`inline-block rounded border px-1.5 py-0.5 text-[10px] font-semibold ${SEV[sev].chip}`}>{SEV[sev].label}</span>
);

const StatusDot = ({ status, eta }) => (
  <div>
    <span className="inline-flex items-center gap-1.5 text-slate-300">
      <span className={`h-2 w-2 rounded-full ${STATUS[status].dot}`} />
      {STATUS[status].label}
    </span>
    {eta != null && <span className="block text-[10px] text-slate-500">ETA {fmtDur(eta)}</span>}
  </div>
);

const Panel = ({ title, sub, right, className = '', children }) => (
  <section className={`flex min-h-0 flex-col overflow-hidden rounded-lg border border-slate-800 bg-slate-900/80 ${className}`}>
    <div className="flex shrink-0 items-center justify-between gap-2 border-b border-slate-800 bg-slate-950/60 px-3 py-2">
      <div className="min-w-0">
        <h2 className="truncate text-sm font-semibold text-slate-200">{title}</h2>
        {sub && <p className="truncate text-[11px] text-slate-500">{sub}</p>}
      </div>
      {right}
    </div>
    {children}
  </section>
);

const Tabs = ({ tabs, value, onChange, label }) => (
  <div role="tablist" aria-label={label} className="flex shrink-0 overflow-x-auto border-b border-slate-800 bg-slate-950/60">
    {tabs.map((t) => (
      <button
        key={t.id} role="tab" aria-selected={value === t.id} onClick={() => onChange(t.id)}
        className={`-mb-px whitespace-nowrap border-b-2 px-3 py-2 text-xs font-medium transition-colors ${focusRing} ${value === t.id ? 'border-red-500 text-slate-100' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
      >
        {t.label}
        {t.count != null && <span className="ml-1.5 rounded-full bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300">{t.count}</span>}
      </button>
    ))}
  </div>
);

const Empty = ({ children }) => <div className="p-6 text-center text-sm text-slate-400">{children}</div>;

const rowKey = (fn) => (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(); } };

// ---------------------------------------------------------------------------
// 7. PANELS
// ---------------------------------------------------------------------------
function IncidentTable({ rows, counts, filter, setFilter, query, setQuery, showClosed, setShowClosed, selectedId, onSelect, started }) {
  const chips = [['ALL', 'All'], ['CRITICAL', 'Critical'], ['HIGH', 'High'], ['MEDIUM', 'Medium'], ['LOW', 'Low']];
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-slate-800 px-3 py-2">
        <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by severity">
          {chips.map(([k, label]) => (
            <button
              key={k} onClick={() => setFilter(k)} aria-pressed={filter === k}
              className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${focusRing} ${filter === k ? 'border-red-500/60 bg-red-500/15 text-red-200' : 'border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200'}`}
            >
              {label} <span className="text-slate-500">{counts[k]}</span>
            </button>
          ))}
        </div>
        <input
          value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search ID, type or area" aria-label="Search incidents"
          className={`ml-auto w-44 rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-200 placeholder:text-slate-600 ${focusRing}`}
        />
        <label className="flex cursor-pointer items-center gap-1.5 text-[11px] text-slate-400">
          <input type="checkbox" checked={showClosed} onChange={(e) => setShowClosed(e.target.checked)} className="accent-red-500" />
          Show closed
        </label>
      </div>
      <div className="agt-scroll min-h-0 flex-1 overflow-auto">
        {rows.length === 0 ? (
          <Empty>
            {!started ? 'No incidents yet. Run the scenario to start the clock.' : (
              <>No incidents match these filters. <button className="text-red-300 underline" onClick={() => { setFilter('ALL'); setQuery(''); }}>Clear filters</button></>
            )}
          </Empty>
        ) : (
          <table className="w-full min-w-[640px] text-left text-xs">
            <thead className="sticky top-0 bg-slate-900 text-[11px] text-slate-500">
              <tr>
                <th className="px-3 py-2 font-medium">Incident</th>
                <th className="py-2 font-medium">Area</th>
                <th className="py-2 font-medium">Severity</th>
                <th className="py-2 font-medium">Status</th>
                <th className="py-2 font-medium">Units</th>
                <th className="py-2 font-medium">Elapsed</th>
                <th className="py-2 pr-3 text-right font-medium">Agent flag</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((i) => (
                <tr
                  key={i.id} tabIndex={0} onClick={() => onSelect(i)} onKeyDown={rowKey(() => onSelect(i))}
                  aria-label={`${i.id}, ${i.type}, ${i.area}`}
                  className={`cursor-pointer border-t border-slate-800/60 transition-colors hover:bg-slate-800/50 ${focusRing} ${selectedId === i.id ? 'bg-slate-800/70' : ''} ${i.status === 'CLOSED' ? 'opacity-50' : ''}`}
                >
                  <td className="px-3 py-2"><span className="font-mono font-semibold text-slate-200">{i.id}</span><span className="block text-[11px] text-slate-400">{i.type}</span></td>
                  <td className="py-2 text-slate-300">{i.area}</td>
                  <td className="py-2"><SevBadge sev={i.sev} /></td>
                  <td className="py-2"><StatusDot status={i.status} eta={i.eta} /></td>
                  <td className="py-2 font-mono text-slate-300">{i.unitTotal}</td>
                  <td className="py-2 font-mono text-slate-400">{fmtElapsed(i.el)}</td>
                  <td className="py-2 pr-3 text-right">
                    {i.flag ? <span className={`inline-block rounded border px-1.5 py-0.5 text-[10px] ${SEV[i.flag.level].chip}`}>{i.flag.text}</span> : <span className="text-slate-600">None</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function StationTable({ stations, onFocus }) {
  const rows = [...stations].sort((a, b) => b.util - a.util);
  return (
    <div className="agt-scroll min-h-0 flex-1 overflow-auto">
      <table className="w-full min-w-[520px] text-left text-xs">
        <thead className="sticky top-0 bg-slate-900 text-[11px] text-slate-500">
          <tr>
            <th className="px-3 py-2 font-medium">Station</th>
            <th className="py-2 font-medium">Incidents</th>
            <th className="py-2 font-medium">Units out</th>
            <th className="py-2 font-medium">Free</th>
            <th className="py-2 pr-3 font-medium">Load</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => (
            <tr key={s.id} tabIndex={0} onClick={() => onFocus(s)} onKeyDown={rowKey(() => onFocus(s))} className={`cursor-pointer border-t border-slate-800/60 hover:bg-slate-800/50 ${focusRing}`}>
              <td className="px-3 py-2 text-slate-200">{s.name}<span className="block text-[10px] text-slate-500">{s.id}</span></td>
              <td className="py-2 font-mono text-slate-300">{s.active}</td>
              <td className="py-2 font-mono text-slate-300">{s.depN}/{s.totalN}</td>
              <td className="py-2 text-[11px] text-slate-400">{UNIT_KEYS.filter((k) => s.free[k] > 0).map((k) => `${s.free[k]} ${UNIT_SHORT[k]}`).join(', ') || 'None'}</td>
              <td className="py-2 pr-3">
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-800"><div className={`h-full ${WORKLOAD[s.workload].bar}`} style={{ width: `${Math.round(s.util * 100)}%` }} /></div>
                  <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${WORKLOAD[s.workload].chip}`}>{WORKLOAD[s.workload].label}</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FleetPanel({ fleet, insights }) {
  return (
    <div className="agt-scroll min-h-0 flex-1 overflow-auto p-3">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="flex flex-col gap-3">
          {UNIT_KEYS.map((k) => {
            const f = fleet[k];
            const pct = (n) => `${(n / f.nominal) * 100}%`;
            return (
              <div key={k}>
                <div className="mb-1 flex justify-between text-xs">
                  <span className="text-slate-300">{UNIT_LABEL[k]}</span>
                  <span className={`font-mono ${f.avail === 0 ? 'text-red-400' : 'text-slate-400'}`}>{f.avail} free of {f.nominal - f.oos}</span>
                </div>
                <div className="flex h-2 overflow-hidden rounded-full bg-slate-800" role="img" aria-label={`${UNIT_LABEL[k]}: ${f.dep} deployed, ${f.avail} free, ${f.oos} out of service`}>
                  <div className="bg-orange-500" style={{ width: pct(f.dep) }} />
                  <div className="bg-emerald-500" style={{ width: pct(f.avail) }} />
                  <div className="bg-slate-600" style={{ width: pct(f.oos) }} />
                </div>
              </div>
            );
          })}
          <div className="flex gap-3 text-[10px] text-slate-500">
            <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-sm bg-orange-500" />Deployed</span>
            <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-sm bg-emerald-500" />Free</span>
            <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-sm bg-slate-600" />Out of service</span>
          </div>
        </div>
        <div>
          <h3 className="mb-2 text-xs font-semibold text-slate-300">What the agent sees</h3>
          <ul className="flex flex-col gap-2.5">
            {insights.map((t, idx) => (
              <li key={idx} className="flex gap-2 text-xs leading-relaxed text-slate-300"><span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400" />{t}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function PriorityQueue({ items, selectedId, onSelect, started }) {
  return (
    <Panel title="Priority queue" sub="Incidents the agent wants you to look at" right={<span className="rounded-full bg-slate-800 px-2 py-0.5 text-[11px] text-slate-300">{items.length}</span>} className="h-96 lg:h-auto lg:flex-1">
      <div className="agt-scroll flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-2">
        {items.length === 0 && <Empty>{started ? 'Nothing needs attention right now.' : 'The queue fills as incidents come in.'}</Empty>}
        {items.map((i) => (
          <button
            key={i.id} onClick={() => onSelect(i)}
            className={`rounded-lg border p-3 text-left transition-colors ${focusRing} ${selectedId === i.id ? 'border-slate-500 bg-slate-800/70' : 'border-slate-800 bg-slate-950/60 hover:border-slate-600'}`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <span className="block font-mono text-sm font-semibold text-slate-100">{i.id}</span>
                <span className="block truncate text-[11px] text-slate-400">{i.type}, {i.area}</span>
              </div>
              <SevBadge sev={i.sev} />
            </div>
            <div className="mt-2 flex items-center gap-2 text-[11px] text-slate-400">
              <span className={`h-2 w-2 rounded-full ${STATUS[i.status].dot}`} />{STATUS[i.status].label}
              <span className="text-slate-600">/</span>{fmtElapsed(i.el)}
              <span className="text-slate-600">/</span>{i.unitTotal} units
            </div>
            <div className={`mt-2 rounded border p-2 ${SEV[i.flag.level].chip}`}>
              <span className="block text-[11px] font-semibold">{i.flag.text}</span>
              <span className="block text-xs leading-snug text-slate-300">{i.flag.reason}</span>
            </div>
          </button>
        ))}
      </div>
    </Panel>
  );
}

function RecsPanel({ recs, recState, setRecState, onSelectId, started }) {
  if (!recs.length) return <Empty>{started ? 'No recommendations. The agent raises them as conditions change.' : 'Recommendations appear once the scenario is running.'}</Empty>;
  return (
    <div className="agt-scroll flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-2">
      {recs.map((r) => {
        const acked = recState[r.id] === 'ACK';
        return (
          <div key={r.id} className={`rounded-lg border border-slate-800 bg-slate-950/60 p-3 ${acked ? 'opacity-70' : ''}`}>
            <div className="flex items-center justify-between gap-2">
              <span className={`flex items-center gap-1.5 text-xs font-semibold ${SEV[r.sev].text}`}><span className={`h-2 w-2 rounded-full ${SEV[r.sev].dot}`} />{r.title}</span>
              <span className="font-mono text-[10px] text-slate-500">{r.clock}</span>
            </div>
            <p className="mt-1 text-xs leading-relaxed text-slate-300">{r.reason}</p>
            <div className="mt-2 flex items-center gap-2">
              {acked ? <span className="text-[11px] font-medium text-emerald-400">Acknowledged</span> : (
                <button onClick={() => setRecState((s) => ({ ...s, [r.id]: 'ACK' }))} className={`rounded border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-300 hover:bg-emerald-500/20 ${focusRing}`}>Acknowledge</button>
              )}
              <button onClick={() => setRecState((s) => ({ ...s, [r.id]: 'DISMISSED' }))} className={`rounded border border-slate-700 px-2 py-0.5 text-[11px] text-slate-400 hover:text-slate-200 ${focusRing}`}>Dismiss</button>
              {r.inc && <button onClick={() => onSelectId(r.inc)} className={`ml-auto text-[11px] text-sky-300 underline ${focusRing}`}>View incident</button>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ActivityPanel({ feed, onSelectId, started }) {
  if (!feed.length) return <Empty>{started ? 'No activity yet.' : 'Events are logged here as the clock runs.'}</Empty>;
  return (
    <ol className="agt-scroll min-h-0 flex-1 overflow-y-auto p-3">
      {feed.map((e, idx) => (
        <li key={`${e.time}-${e.type}-${idx}`} className="flex gap-3 pb-3">
          <span className="w-14 shrink-0 pt-0.5 font-mono text-[11px] text-slate-500">{e.clock}</span>
          <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${SEV[e.sev]?.dot || 'bg-sky-400'}`} />
          <div className="min-w-0 text-xs">
            <span className={`block text-[11px] font-semibold ${SEV[e.sev]?.text || 'text-sky-400'}`}>{e.type}</span>
            {e.id ? <button onClick={() => onSelectId(e.id)} className={`text-left leading-snug text-slate-300 hover:text-white ${focusRing}`}>{e.desc}</button> : <span className="leading-snug text-slate-300">{e.desc}</span>}
          </div>
        </li>
      ))}
    </ol>
  );
}

function Drawer({ inc, open, onClose, onAction, doneActions }) {
  if (!inc) return null;
  const [dispatch, arrive, contain, close] = inc.d;
  const steps = [
    { label: 'Reported', at: 0, done: true },
    { label: 'Units dispatched', at: dispatch, done: inc.el >= dispatch },
    { label: 'First unit on scene', at: arrive, done: inc.el >= arrive, note: inc.el >= arrive ? `Response time ${fmtDur(arrive)}` : inc.eta != null ? `ETA ${fmtDur(inc.eta)}` : '' },
    { label: 'Contained', at: contain, done: inc.el >= contain },
    { label: 'Closed', at: close, done: inc.el >= close },
  ];
  const actions = [
    { key: 'aid', label: 'Request mutual aid', done: 'Mutual aid requested', log: `Mutual aid requested for ${inc.id}.` },
    { key: 'traffic', label: 'Alert traffic control', done: 'Traffic control alerted', log: `Traffic control alerted for ${inc.id}.` },
    { key: 'review', label: 'Mark as reviewed', done: 'Reviewed', log: `${inc.id} marked as reviewed by operator.` },
  ];
  const Section = ({ title, children }) => (
    <section>
      <h3 className="mb-2 border-b border-slate-800 pb-1.5 text-xs font-semibold text-slate-400">{title}</h3>
      {children}
    </section>
  );
  return (
    <aside
      aria-hidden={!open} aria-label="Incident detail"
      className={`fixed bottom-0 right-0 top-16 z-50 flex w-full flex-col border-l border-slate-700 bg-slate-900 shadow-2xl transition-[transform,visibility] duration-300 ease-in-out motion-reduce:transition-none sm:w-[440px] ${open ? 'translate-x-0' : 'invisible translate-x-full'}`}
    >
      <div className="flex shrink-0 items-start justify-between border-b border-slate-800 bg-slate-950/80 p-4">
        <div>
          <p className="text-[11px] text-slate-500">{inc.cat} incident</p>
          <h2 className="font-mono text-xl font-semibold text-slate-100">{inc.id}</h2>
          <p className="mt-0.5 text-sm text-slate-300">{inc.type}</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            <SevBadge sev={inc.sev} />
            <span className="rounded border border-slate-600 bg-slate-800 px-1.5 py-0.5 text-[10px] font-semibold text-slate-300">{STATUS[inc.status].label}</span>
            {inc.priority && <span className="rounded border border-slate-600 bg-slate-800 px-1.5 py-0.5 text-[10px] font-semibold text-slate-300">{inc.priority}</span>}
          </div>
        </div>
        <button onClick={onClose} aria-label="Close incident detail" className={`rounded border border-slate-700 p-1.5 text-slate-400 hover:text-white ${focusRing}`}>
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
        </button>
      </div>

      <div className="agt-scroll flex flex-1 flex-col gap-5 overflow-y-auto p-4">
        {inc.flag && (
          <div className={`rounded-lg border p-3 ${SEV[inc.flag.level].chip}`}>
            <h3 className="flex items-center gap-2 text-xs font-semibold"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current motion-reduce:animate-none" />Agent assessment: {inc.flag.text}</h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-200">{inc.flag.reason}</p>
            <p className="mt-2 border-l-2 border-slate-500 pl-2 text-xs italic leading-relaxed text-slate-300">{inc.flag.action}</p>
          </div>
        )}

        <Section title="Details">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-xs">
            <div><dt className="text-[11px] text-slate-500">Area</dt><dd className="text-slate-200">{inc.area}</dd></div>
            <div><dt className="text-[11px] text-slate-500">Reported</dt><dd className="font-mono text-slate-200">{fmtClock(inc.reportedAbs)}</dd></div>
            <div><dt className="text-[11px] text-slate-500">Elapsed</dt><dd className="font-mono font-semibold text-orange-300">{fmtElapsed(inc.el)}</dd></div>
            <div><dt className="text-[11px] text-slate-500">Primary station</dt><dd className="text-indigo-300">{STN[inc.primary].name}</dd></div>
            <div className="col-span-2"><dt className="text-[11px] text-slate-500">Caller notes</dt><dd className="leading-relaxed text-slate-300">{inc.notes}</dd></div>
          </dl>
        </Section>

        <Section title="Progress">
          <ol className="flex flex-col">
            {steps.map((s, idx) => (
              <li key={s.label} className="flex gap-3">
                <div className="flex flex-col items-center">
                  <span className={`mt-0.5 h-3 w-3 rounded-full border-2 ${s.done ? 'border-emerald-400 bg-emerald-400' : 'border-slate-600 bg-slate-900'}`} />
                  {idx < steps.length - 1 && <span className={`w-px flex-1 ${s.done ? 'bg-emerald-400/50' : 'bg-slate-700'}`} />}
                </div>
                <div className="flex flex-1 items-start justify-between pb-3 text-xs">
                  <div>
                    <span className={s.done ? 'text-slate-200' : 'text-slate-500'}>{s.label}</span>
                    {s.note && <span className="block text-[11px] text-slate-500">{s.note}</span>}
                  </div>
                  <span className="font-mono text-slate-400">{s.done ? fmtClock(inc.reportedAbs + s.at) : '-'}</span>
                </div>
              </li>
            ))}
          </ol>
        </Section>

        <Section title={`Units assigned (${inc.unitTotal})`}>
          <ul className="flex flex-col gap-2">
            {Object.entries(inc.dep).map(([sid, u]) => (
              <li key={sid} className="flex items-center justify-between rounded border border-slate-800 bg-slate-950/60 px-3 py-2 text-xs">
                <span className="text-slate-200">{STN[sid].name}</span>
                <span className="font-mono text-slate-400">{UNIT_KEYS.filter((k) => u[k] > 0).map((k) => `${u[k]} ${UNIT_SHORT[k]}`).join(', ')}</span>
              </li>
            ))}
          </ul>
        </Section>

        {inc.applied.length > 0 && (
          <Section title="Escalation history">
            <ul className="flex flex-col gap-2 text-xs">
              {inc.applied.map((x, idx) => (
                <li key={idx} className="flex gap-3"><span className="w-14 shrink-0 font-mono text-slate-500">{fmtClock(inc.reportedAbs + x.after)}</span><span className="leading-snug text-slate-300">{x.note}</span></li>
              ))}
            </ul>
          </Section>
        )}

        {inc.status !== 'CLOSED' && (
          <Section title="Operator actions">
            <div className="flex flex-wrap gap-2">
              {actions.map((a) => {
                const done = doneActions.has(`${inc.id}:${a.key}`);
                return (
                  <button key={a.key} disabled={done} onClick={() => onAction(inc, a)} className={`rounded border px-3 py-1.5 text-xs font-medium ${focusRing} ${done ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300' : 'border-slate-600 bg-slate-800 text-slate-200 hover:bg-slate-700'}`}>
                    {done ? a.done : a.label}
                  </button>
                );
              })}
            </div>
          </Section>
        )}
        <p className="pb-2 text-center text-[11px] text-slate-600">Simulated incident data. Actions are logged in the simulation only.</p>
      </div>
    </aside>
  );
}

// ---------------------------------------------------------------------------
// 8. MAIN COMPONENT
// ---------------------------------------------------------------------------
export default function Agent() {
  const [scId, setScId] = useState(SCENARIOS[0].id);
  const sc = useMemo(() => SCENARIOS.find((s) => s.id === scId), [scId]);

  const [t, setT] = useState(0);
  const tRef = useRef(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(5);
  const [runKey, setRunKey] = useState(0);

  const [selectedId, setSelectedId] = useState(null);
  const [focus, setFocus] = useState(null);
  const [filter, setFilter] = useState('ALL');
  const [query, setQuery] = useState('');
  const [showClosed, setShowClosed] = useState(false);
  const [showStations, setShowStations] = useState(true);
  const [leftTab, setLeftTab] = useState('incidents');
  const [rightTab, setRightTab] = useState('recs');
  const [recState, setRecState] = useState({});
  const [actions, setActions] = useState(() => new Set());
  const [oplog, setOplog] = useState([]);
  const [toast, setToast] = useState(null);

  const tSec = Math.floor(t);
  const started = tSec > 0 || running;

  // ---- clock ----
  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(() => {
      const n = Math.min(tRef.current + speed * 0.25, sc.duration);
      tRef.current = n;
      setT(n);
      if (n >= sc.duration) setRunning(false);
    }, 250);
    return () => clearInterval(id);
  }, [running, speed, sc]);

  const seek = (v) => { tRef.current = v; setT(v); };
  const loadScenario = useCallback((id) => {
    setScId(id); setRunning(false); tRef.current = 0; setT(0);
    setSelectedId(null); setRecState({}); setActions(new Set()); setOplog([]); setRunKey((k) => k + 1);
  }, []);
  const reset = useCallback(() => loadScenario(scId), [loadScenario, scId]);
  const toggleRun = useCallback(() => {
    if (tRef.current >= sc.duration) { seek(0); setRunning(true); return; }
    setRunning((r) => !r);
  }, [sc]);

  // ---- keyboard shortcuts ----
  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.code === 'Space' && tag !== 'BUTTON') { e.preventDefault(); toggleRun(); }
      else if (e.key === 'r' || e.key === 'R') reset();
      else if (e.key === 'Escape') setSelectedId(null);
      else if (/^[1-5]$/.test(e.key)) setSpeed(SPEEDS[Number(e.key) - 1]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggleRun, reset]);

  useEffect(() => {
    if (!toast) return undefined;
    const id = setTimeout(() => setToast(null), 2600);
    return () => clearTimeout(id);
  }, [toast]);

  // ---- derived state ----
  const state = useMemo(() => computeState(sc, tSec), [sc, tSec]);
  const aid = useMemo(() => new Set([...actions].filter((k) => k.endsWith(':aid')).map((k) => k.split(':')[0])), [actions]);
  const incidents = useMemo(() => state.incidents.map((i) => ({ ...i, flag: deriveFlag(i, state.stnById, aid) })), [state, aid]);
  const insights = useMemo(() => buildInsights(state), [state]);
  const allEvents = useMemo(() => buildEvents(sc), [sc]);

  const clockAbs = sc.clock + tSec;
  const feed = useMemo(
    () => [...oplog, ...allEvents].filter((e) => e.time <= tSec).sort((a, b) => b.time - a.time).slice(0, 40).map((e) => ({ ...e, clock: fmtClock(sc.clock + e.time) })),
    [allEvents, oplog, tSec, sc],
  );
  const recs = useMemo(
    () => sc.recs.filter((r) => r.at <= tSec && recState[r.id] !== 'DISMISSED').sort((a, b) => b.at - a.at).map((r) => ({ ...r, clock: fmtClock(sc.clock + r.at) })),
    [sc, tSec, recState],
  );
  const pendingRecs = recs.filter((r) => recState[r.id] !== 'ACK').length;

  const visible = incidents.filter((i) => showClosed || i.status !== 'CLOSED');
  const counts = { ALL: visible.length, CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
  visible.forEach((i) => { counts[i.sev] += 1; });
  const q = query.trim().toLowerCase();
  const rows = visible
    .filter((i) => (filter === 'ALL' || i.sev === filter) && (!q || `${i.id} ${i.type} ${i.area}`.toLowerCase().includes(q)))
    .sort(byPriority);
  const queue = incidents.filter((i) => i.flag).sort(byPriority);
  const selected = incidents.find((i) => i.id === selectedId) || null;
  const lastSel = useRef(null);
  if (selected) lastSel.current = selected;

  const select = (i) => { setSelectedId(i.id); setFocus({ lat: i.lat, lng: i.lng, zoom: 13, k: Date.now() }); };
  const selectById = (id) => { const i = incidents.find((x) => x.id === id); if (i) select(i); };
  const focusStation = (s) => setFocus({ lat: s.lat, lng: s.lng, zoom: 13, k: Date.now() });

  const onAction = (inc, a) => {
    setActions((prev) => new Set(prev).add(`${inc.id}:${a.key}`));
    setOplog((prev) => [...prev, { id: inc.id, time: tSec, type: 'OPERATOR', sev: 'INFO', desc: a.log }]);
    setToast(a.log);
  };

  const ticks = useMemo(() => {
    const out = [];
    sc.incidents.forEach((i) => {
      if (i.cat === 'Standby') return;
      if (i.at > 0) out.push({ at: i.at, sev: i.sev, label: `${fmtClock(sc.clock + i.at)} ${i.type}, ${i.area}` });
      (i.esc || []).forEach((x) => { const at = i.at + x.after; if (at > 0 && at <= sc.duration) out.push({ at, sev: x.sev, label: `${fmtClock(sc.clock + at)} Escalation, ${i.area}` }); });
    });
    return out;
  }, [sc]);

  const k = state.kpis;
  const overTarget = k.avgResp != null && k.avgResp > RESPONSE_TARGET;
  const kpiTiles = [
    { label: 'Active incidents', val: k.active, sub: k.newLast10 ? `${k.newLast10} new in the last 10 min` : 'None new in the last 10 min', tone: 'text-slate-50' },
    { label: 'Critical', val: k.critical, sub: `${k.criticalOnScene} on scene`, tone: 'text-red-400' },
    { label: 'High priority', val: k.high, sub: k.high ? 'Watch for escalation' : 'None active', tone: 'text-orange-400' },
    { label: 'Units deployed', val: k.deployed, sub: `of ${k.inService} in service`, tone: 'text-indigo-300' },
    { label: 'Units on scene', val: k.onScene, sub: `${k.enRoute} still en route`, tone: 'text-sky-300' },
    { label: 'Stations under load', val: k.underLoad, sub: `of ${STATIONS.length} stations`, tone: 'text-yellow-300' },
    { label: 'Avg response', val: k.avgResp == null ? '-' : fmtDur(k.avgResp), sub: k.avgResp == null ? 'No arrivals yet' : overTarget ? `${fmtDur(k.avgResp - RESPONSE_TARGET)} over ${fmtDur(RESPONSE_TARGET)} target` : `Within ${fmtDur(RESPONSE_TARGET)} target`, tone: overTarget ? 'text-red-400' : 'text-emerald-300' },
  ];

  const runLabel = t >= sc.duration ? 'Replay' : running ? 'Pause' : t > 0 ? 'Resume' : 'Run scenario';
  const clockNote = t >= sc.duration ? 'Scenario complete' : running ? `Running at ${speed}x` : t > 0 ? 'Paused' : 'Ready';

  return (
    <div className="relative flex min-h-[calc(100vh-64px)] flex-col gap-3 overflow-y-auto overflow-x-hidden bg-slate-950 p-3 font-sans text-slate-100 sm:p-4 lg:h-[calc(100vh-64px)] lg:overflow-hidden">
      <style>{`
        .agt-marker{background:transparent;border:0}
        .agt-dot{position:relative;border-radius:50%;box-sizing:border-box}
        .agt-pulse::after{content:'';position:absolute;inset:-6px;border-radius:50%;border:2px solid var(--c);opacity:.6;animation:agt-ping 1.8s ease-out infinite}
        @keyframes agt-ping{0%{transform:scale(.6);opacity:.7}100%{transform:scale(1.8);opacity:0}}
        .agt-tip.leaflet-tooltip{background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:6px;box-shadow:0 4px 14px rgba(0,0,0,.5);padding:6px 8px;font-size:11px}
        .agt-tip.leaflet-tooltip-top:before{border-top-color:#334155}
        .agt-scroll{scrollbar-width:thin;scrollbar-color:#334155 transparent}
        .agt-scroll::-webkit-scrollbar{width:6px;height:6px}
        .agt-scroll::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}
        @media (prefers-reduced-motion:reduce){.agt-pulse::after{animation:none}}
      `}</style>

      {/* HEADER */}
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-xl font-semibold tracking-tight text-slate-50">Fire operations agent</h1>
            <Chip tone="emerald" pulse>System operational</Chip>
            <Chip tone="amber">Simulated data</Chip>
          </div>
          <p className="mt-1 max-w-2xl text-xs text-slate-400">The agent ranks incidents by severity, elapsed time and station workload. It recommends; the operator decides.</p>
        </div>
        <div className="text-right" aria-live="off">
          <div className="font-mono text-3xl font-semibold tabular-nums text-slate-50">{fmtClock(clockAbs)}</div>
          <div className="text-[11px] text-slate-500">{clockNote}</div>
        </div>
      </header>

      {/* SCENARIO + SIMULATION CONTROLS */}
      <section className="shrink-0 rounded-lg border border-slate-800 bg-slate-900/80 p-3" aria-label="Simulation controls">
        <div role="tablist" aria-label="Scenarios" className="flex flex-wrap gap-2">
          {SCENARIOS.map((s) => (
            <button
              key={s.id} role="tab" aria-selected={s.id === scId} onClick={() => loadScenario(s.id)}
              className={`rounded-md border px-3 py-1.5 text-left transition-colors ${focusRing} ${s.id === scId ? 'border-red-500/60 bg-red-500/10 text-slate-50' : 'border-slate-700 text-slate-300 hover:border-slate-500'}`}
            >
              <span className="block text-xs font-semibold">{s.name}</span>
              <span className="block text-[10px] text-slate-500">{s.tag}</span>
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs leading-relaxed text-slate-400">{sc.blurb}</p>

        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            onClick={toggleRun}
            className={`min-w-[132px] rounded-md px-4 py-2 text-sm font-semibold ${focusRing} ${running ? 'border border-amber-500/50 bg-amber-500/15 text-amber-300' : 'bg-red-600 text-white hover:bg-red-500'}`}
          >
            {running ? 'Pause' : runLabel}
          </button>
          <button onClick={reset} className={`rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-200 hover:bg-slate-700 ${focusRing}`}>Reset</button>

          <div role="radiogroup" aria-label="Simulation speed" className="flex overflow-hidden rounded-md border border-slate-700">
            {SPEEDS.map((s, idx) => (
              <button
                key={s} role="radio" aria-checked={speed === s} onClick={() => setSpeed(s)} title={`Speed ${s}x (press ${idx + 1})`}
                className={`px-3 py-2 font-mono text-xs ${focusRing} ${speed === s ? 'bg-slate-100 font-semibold text-slate-900' : 'bg-slate-900 text-slate-300 hover:bg-slate-800'}`}
              >
                {s}x
              </button>
            ))}
          </div>

          <div className="relative h-9 min-w-[220px] flex-1">
            <div className="absolute inset-x-1 top-0 h-3">
              {ticks.map((tk, idx) => (
                <span key={idx} title={tk.label} className={`absolute top-0 h-3 w-0.5 -translate-x-1/2 ${SEV[tk.sev].dot}`} style={{ left: `${(tk.at / sc.duration) * 100}%` }} />
              ))}
            </div>
            <input
              type="range" min={0} max={sc.duration} step={1} value={tSec} onChange={(e) => seek(Number(e.target.value))}
              aria-label="Scenario timeline" aria-valuetext={`${fmtDur(tSec)} of ${fmtDur(sc.duration)}`}
              className="absolute bottom-0 w-full accent-red-500"
            />
          </div>
          <div className="font-mono text-xs tabular-nums text-slate-400">T+{fmtDur(tSec)} / {fmtDur(sc.duration)}</div>
        </div>
        <p className="mt-2 text-[11px] text-slate-600">Shortcuts: Space run or pause, R reset, 1 to 5 speed, Esc close detail. Drag the timeline to jump; ticks mark new incidents and escalations.</p>
      </section>

      {/* KPI STRIP */}
      <div className="grid shrink-0 grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-7">
        {kpiTiles.map((c) => (
          <div key={c.label} className="rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-2">
            <span className="text-[11px] text-slate-500">{c.label}</span>
            <div className={`font-mono text-2xl font-semibold tabular-nums ${c.tone}`}>{c.val}</div>
            <span className="block truncate text-[11px] text-slate-500">{c.sub}</span>
          </div>
        ))}
      </div>

      {/* MAIN LAYOUT */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 lg:grid-cols-3">
        {/* LEFT: map + tabs */}
        <div className="flex min-h-0 flex-col gap-3 lg:col-span-2">
          <div className="relative isolate h-80 overflow-hidden rounded-lg border border-slate-800 lg:h-auto lg:flex-[1.15]">
            <div className="absolute left-3 top-3 z-[500] flex items-center gap-3 rounded-md border border-slate-700 bg-slate-950/90 px-2.5 py-1.5 text-[11px] text-slate-300">
              <span className="font-semibold">Incident map</span>
              <label className="flex cursor-pointer items-center gap-1.5 text-slate-400">
                <input type="checkbox" checked={showStations} onChange={(e) => setShowStations(e.target.checked)} className="accent-red-500" />
                Stations
              </label>
            </div>
            <div className="absolute bottom-3 left-3 z-[500] flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-slate-700 bg-slate-950/90 px-2.5 py-1.5 text-[10px] text-slate-400">
              {['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map((s) => <span key={s} className="flex items-center gap-1"><i className={`h-2 w-2 rounded-full ${SEV[s].dot}`} />{SEV[s].label}</span>)}
              <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-full border border-dashed border-slate-300" />Responding</span>
              <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-full bg-slate-500" />Contained</span>
              <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-sm bg-emerald-500" />Station</span>
            </div>
            <MapContainer center={sc.center} zoom={sc.zoom} style={{ height: '100%', width: '100%' }} zoomControl={false}>
              <MapController view={{ key: `${sc.id}-${runKey}`, center: sc.center, zoom: sc.zoom }} focus={focus} />
               <TileLayer
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                      />
              {showStations && state.stations.map((s) => (
                <Marker key={`${s.id}-${s.workload}`} position={[s.lat, s.lng]} icon={stationIcon(s.workload)} eventHandlers={{ click: () => { setLeftTab('stations'); } }}>
                  <Tooltip direction="top" offset={[0, -6]} className="agt-tip">
                    <strong>{s.name} station</strong><br />{s.depN} of {s.totalN} units out, {WORKLOAD[s.workload].label.toLowerCase()}
                  </Tooltip>
                </Marker>
              ))}
              {incidents.filter((i) => i.status !== 'CLOSED' || showClosed).map((i) => (
                <Marker
                  key={`${i.id}-${i.sev}-${i.status}-${selectedId === i.id}`} position={[i.lat, i.lng]}
                  icon={incidentIcon(i.sev, i.status, selectedId === i.id)} zIndexOffset={1000 - SEV[i.sev].rank * 100}
                  eventHandlers={{ click: () => select(i) }}
                >
                  <Tooltip direction="top" offset={[0, -8]} className="agt-tip">
                    <strong>{i.id}</strong><br />{i.type}, {i.area}<br />{SEV[i.sev].label}, {STATUS[i.status].label.toLowerCase()}
                  </Tooltip>
                </Marker>
              ))}
            </MapContainer>
          </div>

          <Panel title="Operations" className="h-96 lg:h-auto lg:flex-1">
            <Tabs
              label="Operations views" value={leftTab} onChange={setLeftTab}
              tabs={[
                { id: 'incidents', label: 'Incidents', count: counts.ALL },
                { id: 'stations', label: 'Stations', count: state.stations.length },
                { id: 'fleet', label: 'Fleet and insights' },
              ]}
            />
            {leftTab === 'incidents' && (
              <IncidentTable
                rows={rows} counts={counts} filter={filter} setFilter={setFilter} query={query} setQuery={setQuery}
                showClosed={showClosed} setShowClosed={setShowClosed} selectedId={selectedId} onSelect={select} started={started}
              />
            )}
            {leftTab === 'stations' && <StationTable stations={state.stations} onFocus={focusStation} />}
            {leftTab === 'fleet' && <FleetPanel fleet={state.fleet} insights={insights} />}
          </Panel>
        </div>

        {/* RIGHT: queue + recommendations/activity */}
        <div className="flex min-h-0 flex-col gap-3">
          <PriorityQueue items={queue} selectedId={selectedId} onSelect={select} started={started} />
          <Panel title="Agent log" className="h-80 shrink-0 lg:h-72">
            <Tabs
              label="Agent log views" value={rightTab} onChange={setRightTab}
              tabs={[
                { id: 'recs', label: 'Recommendations', count: pendingRecs },
                { id: 'activity', label: 'Activity' },
              ]}
            />
            {rightTab === 'recs'
              ? <RecsPanel recs={recs} recState={recState} setRecState={setRecState} onSelectId={selectById} started={started} />
              : <ActivityPanel feed={feed} onSelectId={selectById} started={started} />}
          </Panel>
        </div>
      </div>

      <Drawer inc={selected || lastSel.current} open={!!selected} onClose={() => setSelectedId(null)} onAction={onAction} doneActions={actions} />

      {toast && (
        <div role="status" className="fixed bottom-4 left-1/2 z-[60] -translate-x-1/2 rounded-md border border-slate-600 bg-slate-800 px-4 py-2 text-xs text-slate-100 shadow-xl">
          {toast}
        </div>
      )}
    </div>
  );
}