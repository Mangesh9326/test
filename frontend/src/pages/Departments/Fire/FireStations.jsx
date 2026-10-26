// frontend/src/pages/Departments/Fire/FireStations.jsx
import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, ZoomControl, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { fireSim } from './fireStationsMock';

// ---------- Constants & utilities ----------
// 'sim' = realistic simulated Mumbai Fire Brigade data (default). 'api' = your /api/fire/stations endpoints.
const DEFAULT_SOURCE = import.meta.env?.VITE_FIRE_DATA_SOURCE === 'api' ? 'api' : 'sim';
const REFRESH_MS = { sim: 5000, api: 15000 };
const STATUS_RANK = { 'HIGH LOAD': 0, BUSY: 1, OPERATIONAL: 2 };
const STEPS = ['DISPATCHED', 'EN_ROUTE', 'ON_SCENE', 'CONTAINED'];

// A station is at HIGH LOAD when it has 3+ active incidents, or when it has run out of free engines
const getDerivedStatus = (s) => {
  const n = Number(s?.active_incidents) || 0;
  if (n === 0) return 'OPERATIONAL';
  if (s.engines > 0 && s.engines_available === 0) return 'HIGH LOAD';
  return n <= 2 ? 'BUSY' : 'HIGH LOAD';
};

const StatusColors = {
  OPERATIONAL: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
  BUSY: 'text-orange-400 bg-orange-400/10 border-orange-400/30',
  'HIGH LOAD': 'text-red-500 bg-red-500/10 border-red-500/30',
};
const MarkerColors = { OPERATIONAL: '#22c55e', BUSY: '#f97316', 'HIGH LOAD': '#ef4444' };
const IncidentHex = { DISPATCHED: '#818cf8', EN_ROUTE: '#c084fc', ON_SCENE: '#fb923c', CONTAINED: '#facc15' };
const IncidentText = { DISPATCHED: 'text-indigo-400', EN_ROUTE: 'text-purple-400', ON_SCENE: 'text-orange-400', CONTAINED: 'text-yellow-400' };
const EventDot = { NEW: 'bg-red-500', EN_ROUTE: 'bg-purple-400', ON_SCENE: 'bg-orange-400', CONTAINED: 'bg-yellow-400', CLEARED: 'bg-emerald-400' };

const hasCoords = (s) => Number.isFinite(s?.latitude) && Number.isFinite(s?.longitude);
const pretty = (v) => String(v ?? '').replace(/_/g, ' ');

const normalizeStation = (s) => ({
  ...s,
  latitude: Number(s.latitude),
  longitude: Number(s.longitude),
  active_incidents: Number(s.active_incidents) || 0,
  engines: Number(s.metadata?.engines) || 0,
  engines_available: s.engines_available != null ? Number(s.engines_available) : null,
  ladders: s.metadata?.ladders ?? null,
  crew: s.metadata?.crew ?? null,
  zone: s.zone || s.metadata?.zone || null,
  avg_response_time: s.avg_response_time != null && s.avg_response_time !== '' ? Number(s.avg_response_time) : null,
});

const getJSON = async (url, signal) => {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${url} responded with HTTP ${res.status}`);
  return res.json();
};

// Same method names as fireSim, so the page doesn't care where data comes from
const liveApi = {
  stations: (s) => getJSON('/api/fire/stations', s),
  summary: (s) => getJSON('/api/fire/stations/summary', s),
  resources: (s) => getJSON('/api/fire/stations/resources', s),
  station: (id, s) => getJSON(`/api/fire/stations/${id}`, s),
  stationIncidents: (id, s) => getJSON(`/api/fire/stations/${id}/incidents`, s),
  activeIncidents: async () => [],
  events: async () => [],
};

const timeAgo = (ts) => {
  if (!ts) return '—';
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 5) return 'just now';
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
};

// Marker icons are cached so react-leaflet doesn't rebuild every marker on each refresh
const iconCache = {};
const stationIcon = (status, selected) => {
  const key = `s-${status}-${selected}`;
  if (!iconCache[key]) {
    const c = MarkerColors[status] || '#3b82f6';
    const size = selected ? 22 : 16;
    iconCache[key] = L.divIcon({
      className: 'custom-station-marker',
      html: `<div class="${status === 'HIGH LOAD' ? 'fs-pulse' : ''}" style="color:${c};background:${c};width:${size}px;height:${size}px;border-radius:50%;border:2px solid ${selected ? '#fff' : '#cbd5e1'};box-shadow:0 0 ${selected ? 16 : 10}px ${c};"></div>`,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });
  }
  return iconCache[key];
};
const incidentIcon = (status, highlighted) => {
  const key = `i-${status}-${highlighted}`;
  if (!iconCache[key]) {
    const c = IncidentHex[status] || '#f97316';
    const size = highlighted ? 16 : 12;
    iconCache[key] = L.divIcon({
      className: 'custom-incident-marker',
      html: `<div class="${status === 'ON_SCENE' ? 'fs-pulse' : ''}" style="color:${c};background:${c};width:${size}px;height:${size}px;transform:rotate(45deg);border:2px solid ${highlighted ? '#fff' : '#0f172a'};"></div>`,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });
  }
  return iconCache[key];
};

// Fits the map to the visible stations only when the set of stations changes (not on every refresh),
// and flies to a station when one is selected.
const MapController = ({ stations, selected }) => {
  const map = useMap();
  const idsKey = stations.map((s) => s.entity_id).join('|');
  const firstFit = useRef(true);
  const selectedId = selected?.entity_id;

  useEffect(() => {
    if (selectedId) return;
    const pts = stations.filter(hasCoords).map((s) => [s.latitude, s.longitude]);
    if (pts.length) {
      map.fitBounds(L.latLngBounds(pts), { padding: [50, 50], maxZoom: 14, animate: !firstFit.current });
      firstFit.current = false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey, selectedId, map]);

  useEffect(() => {
    if (selectedId && hasCoords(selected)) {
      map.flyTo([selected.latitude, selected.longitude], Math.max(map.getZoom(), 14), { duration: 1.2 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, map]);

  return null;
};

const SortHeader = ({ label, k, sort, onSort, className = '' }) => (
  <button
    type="button"
    onClick={() => onSort(k)}
    className={`flex items-center gap-1 uppercase font-bold hover:text-slate-200 transition-colors ${className}`}
  >
    {label}
    <span className={sort.key === k ? 'text-red-400' : 'text-slate-700'}>
      {sort.key === k ? (sort.dir === 'asc' ? '▲' : '▼') : '↕'}
    </span>
  </button>
);

const EMPTY_DETAIL = { station: null, incidents: [], loading: false, error: null };

export default function FireStations() {
  const [source, setSource] = useState(DEFAULT_SOURCE);
  const api = source === 'sim' ? fireSim : liveApi;
  const refreshMs = REFRESH_MS[source];

  const [stations, setStations] = useState([]);
  const [summary, setSummary] = useState({});
  const [resources, setResources] = useState({});
  const [liveIncidents, setLiveIncidents] = useState([]);
  const [activity, setActivity] = useState([]);
  const [toasts, setToasts] = useState([]);
  const [lowerTab, setLowerTab] = useState('workload');
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(EMPTY_DETAIL);

  const [filters, setFilters] = useState({ search: '', status: 'ALL', active_only: false });
  const [sort, setSort] = useState({ key: 'status', dir: 'asc' });

  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [, setTick] = useState(0);
  const abortRef = useRef(null);
  const lastEventRef = useRef(0);
  const searchRef = useRef(null);

  const pushToast = useCallback((ev) => {
    setToasts((t) => [...t.slice(-2), ev]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== ev.id)), 7000);
  }, []);

  // ---------- Data loading ----------
  const loadAll = useCallback(async () => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setIsRefreshing(true);
    try {
      const [st, sum, res, act, ev] = await Promise.allSettled([
        api.stations(ctrl.signal),
        api.summary(ctrl.signal),
        api.resources(ctrl.signal),
        api.activeIncidents(ctrl.signal),
        api.events(lastEventRef.current),
      ]);
      if (ctrl.signal.aborted) return;
      if (st.status === 'rejected') throw st.reason;
      if (!Array.isArray(st.value)) throw new Error('Unexpected response from /api/fire/stations');

      setStations(st.value.map(normalizeStation));
      if (sum.status === 'fulfilled' && sum.value && !sum.value.error) setSummary(sum.value);
      if (res.status === 'fulfilled' && res.value && !res.value.error) setResources(res.value);
      if (act.status === 'fulfilled' && Array.isArray(act.value)) setLiveIncidents(act.value);

      if (ev.status === 'fulfilled' && Array.isArray(ev.value) && ev.value.length) {
        const fresh = [...ev.value].sort((a, b) => b.id - a.id);
        const firstLoad = lastEventRef.current === 0;
        lastEventRef.current = fresh[0].id;
        setActivity((prev) => [...fresh, ...prev].slice(0, 40));
        if (!firstLoad) fresh.filter((e) => e.kind === 'NEW').slice(0, 2).forEach(pushToast);
      }
      setError(null);
      setLastUpdated(Date.now());
    } catch (err) {
      if (err.name === 'AbortError') return;
      console.error('Error fetching station data:', err);
      setError(err.message || 'Failed to load station data.');
    } finally {
      if (!ctrl.signal.aborted) {
        setIsLoading(false);
        setIsRefreshing(false);
      }
    }
  }, [api, pushToast]);

  useEffect(() => {
    loadAll();
    return () => abortRef.current?.abort();
  }, [loadAll]);

  useEffect(() => {
    if (!autoRefresh) return undefined;
    const t = setInterval(() => { if (!document.hidden) loadAll(); }, refreshMs);
    return () => clearInterval(t);
  }, [autoRefresh, loadAll, refreshMs]);

  // keeps the "updated Xs ago" labels moving
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 5000);
    return () => clearInterval(t);
  }, []);

  const switchSource = (next) => {
    if (next === source) return;
    abortRef.current?.abort();
    lastEventRef.current = 0;
    setStations([]);
    setLiveIncidents([]);
    setActivity([]);
    setToasts([]);
    setSelectedId(null);
    setError(null);
    setLastUpdated(null);
    setIsLoading(true);
    setSource(next);
  };

  // Station detail + assigned incidents; refreshed with the page while the drawer is open
  useEffect(() => {
    if (!selectedId) { setDetail(EMPTY_DETAIL); return undefined; }
    const ctrl = new AbortController();
    const run = async (silent) => {
      if (!silent) setDetail({ station: null, incidents: [], loading: true, error: null });
      try {
        const [d, inc] = await Promise.all([
          api.station(selectedId, ctrl.signal),
          api.stationIncidents(selectedId, ctrl.signal),
        ]);
        if (ctrl.signal.aborted) return;
        setDetail({
          station: d && !d.error ? d : null,
          incidents: Array.isArray(inc) ? inc : [],
          loading: false,
          error: null,
        });
      } catch (err) {
        if (err.name === 'AbortError') return;
        console.error('Error fetching station details', err);
        setDetail((prev) => ({ ...prev, loading: false, error: 'Live details could not be loaded for this station.' }));
      }
    };
    run(false);
    const t = autoRefresh ? setInterval(() => { if (!document.hidden) run(true); }, refreshMs) : null;
    return () => { ctrl.abort(); if (t) clearInterval(t); };
  }, [selectedId, autoRefresh, api, refreshMs]);

  // Keyboard: Esc closes the drawer, "/" jumps to search
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') setSelectedId(null);
      const typing = ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName);
      if (e.key === '/' && !typing) { e.preventDefault(); searchRef.current?.focus(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // ---------- Derived data ----------
  const selectedStation = useMemo(() => {
    const base = stations.find((s) => s.entity_id === selectedId);
    if (!base) return null;
    const d = detail.station ? normalizeStation(detail.station) : {};
    return { ...base, ...d, latitude: base.latitude, longitude: base.longitude };
  }, [stations, selectedId, detail.station]);

  // KPI counts come from the same list as the table and map, so the numbers always agree
  const kpis = useMemo(() => {
    const counts = { OPERATIONAL: 0, BUSY: 0, 'HIGH LOAD': 0 };
    let active = 0, engines = 0, free = 0, freeKnown = false;
    const times = [];
    stations.forEach((s) => {
      counts[getDerivedStatus(s)] += 1;
      active += s.active_incidents;
      engines += s.engines;
      if (s.engines_available != null) { free += s.engines_available; freeKnown = true; }
      if (s.avg_response_time > 0) times.push(s.avg_response_time);
    });
    const avg = times.length ? times.reduce((a, b) => a + b, 0) / times.length : Number(summary.averageResponseTime) || null;
    const totalEngines = resources.totalFireEngines ?? (engines || null);
    return { total: stations.length, ...counts, active, engines: totalEngines, free: freeKnown ? free : null, avg };
  }, [stations, summary, resources]);

  const rows = useMemo(() => {
    const q = filters.search.trim().toLowerCase();
    const list = stations.filter((s) => {
      const status = getDerivedStatus(s);
      return (!q || (s.name || '').toLowerCase().includes(q) || (s.zone || '').toLowerCase().includes(q))
        && (filters.status === 'ALL' || status === filters.status)
        && (!filters.active_only || s.active_incidents > 0);
    });
    const val = {
      name: (s) => s.name || '',
      status: (s) => STATUS_RANK[getDerivedStatus(s)],
      incidents: (s) => s.active_incidents,
      engines: (s) => s.engines_available ?? s.engines,
      response: (s) => s.avg_response_time ?? 1e9,
    }[sort.key];
    const dir = sort.dir === 'asc' ? 1 : -1;
    return [...list].sort((a, b) => {
      const x = val(a), y = val(b);
      const r = typeof x === 'string' ? x.localeCompare(y) : x - y;
      return r ? r * dir : (b.active_incidents - a.active_incidents || (a.name || '').localeCompare(b.name || ''));
    });
  }, [stations, filters, sort]);

  const mapStations = useMemo(() => rows.filter(hasCoords), [rows]);
  const topWorkload = useMemo(
    () => [...stations].filter((s) => s.active_incidents > 0).sort((a, b) => b.active_incidents - a.active_incidents).slice(0, 4),
    [stations],
  );
  const maxLoad = topWorkload[0]?.active_incidents || 1;
  const filtersActive = filters.search || filters.status !== 'ALL' || filters.active_only;
  const onSort = (key) => setSort((s) => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: key === 'incidents' ? 'desc' : 'asc' }));

  const kpiCards = [
    { label: 'TOTAL STATIONS', value: kpis.total, color: 'text-slate-200' },
    { label: 'OPERATIONAL', value: kpis.OPERATIONAL, color: 'text-emerald-400' },
    { label: 'BUSY', value: kpis.BUSY, color: 'text-orange-400' },
    { label: 'HIGH LOAD', value: kpis['HIGH LOAD'], color: 'text-red-500' },
    { label: 'ACTIVE INCIDENTS', value: kpis.active, color: 'text-blue-400' },
    kpis.free != null
      ? { label: 'ENGINES FREE', value: `${kpis.free}/${kpis.engines ?? '—'}`, color: 'text-slate-300' }
      : { label: 'TOTAL ENGINES', value: kpis.engines, color: 'text-slate-300' },
    { label: 'AVG RESPONSE', value: kpis.avg ? `${kpis.avg.toFixed(1)}m` : null, color: 'text-slate-300' },
  ];

  const statusPills = [
    { key: 'ALL', label: 'All', count: kpis.total },
    { key: 'OPERATIONAL', label: 'Operational', count: kpis.OPERATIONAL },
    { key: 'BUSY', label: 'Busy', count: kpis.BUSY },
    { key: 'HIGH LOAD', label: 'High load', count: kpis['HIGH LOAD'] },
  ];

  const sel = selectedStation;
  const selStatus = sel ? getDerivedStatus(sel) : null;
  const stepIdx = (status) => STEPS.indexOf(status);

  return (
    <div className="flex flex-col min-h-[calc(100vh-64px)] lg:h-[calc(100vh-64px)] bg-slate-950 text-slate-100 p-4 gap-4 lg:overflow-hidden font-sans">
      <style>{`
        @keyframes fs-pulse { 0% { box-shadow: 0 0 0 0 currentColor; } 100% { box-shadow: 0 0 0 16px transparent; } }
        .fs-pulse { animation: fs-pulse 1.6s ease-out infinite; }
        @keyframes fs-toast-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
        .fs-toast { animation: fs-toast-in .25s ease-out; }
        @media (prefers-reduced-motion: reduce) { .fs-pulse, .fs-toast { animation: none; } }
        .fs-popup .leaflet-popup-content { margin: 0; }
      `}</style>

      {/* HEADER & KPIs */}
      <div className="shrink-0 flex flex-col gap-4">
        <div className="flex flex-col md:flex-row md:justify-between md:items-end gap-3 border-b border-slate-800 pb-3">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-red-500">FIRE STATIONS</h1>
            <p className="text-xs text-slate-400 font-mono mt-1">
              {source === 'sim' ? 'MUMBAI FIRE BRIGADE · SIMULATED CONTROL ROOM VIEW' : 'DERIVED FROM CITY MAP ENTITIES'}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex rounded-md overflow-hidden border border-slate-700 text-[10px] font-mono" role="group" aria-label="Data source">
              {[['sim', 'Simulation'], ['api', 'Live API']].map(([k, label]) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => switchSource(k)}
                  aria-pressed={source === k}
                  className={`px-3 py-1.5 transition-colors ${source === k ? 'bg-red-600 text-white' : 'bg-slate-900 text-slate-400 hover:text-slate-200'}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="text-[10px] font-mono bg-red-950/40 text-red-400 border border-red-900/50 px-3 py-1.5 rounded-md uppercase flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
              {source === 'sim' ? 'SIMULATED OPERATIONAL DATA' : 'DEMO / SIMULATED OPERATIONAL DATA'}
            </div>
            <div className="text-[10px] font-mono bg-slate-900 text-slate-400 border border-slate-800 px-3 py-1.5 rounded-md flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${error ? 'bg-amber-400' : autoRefresh ? 'bg-emerald-400' : 'bg-slate-600'}`}></span>
              Updated {timeAgo(lastUpdated)}
            </div>
            <label className="flex items-center gap-1.5 text-[10px] font-mono text-slate-400 cursor-pointer">
              <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} className="accent-red-500" />
              Auto-refresh ({refreshMs / 1000}s)
            </label>
            <button
              type="button"
              onClick={loadAll}
              disabled={isRefreshing}
              className="text-[10px] font-mono bg-slate-900 hover:bg-slate-800 disabled:opacity-50 border border-slate-700 text-slate-200 px-3 py-1.5 rounded-md transition-colors"
            >
              {isRefreshing ? 'Refreshing…' : 'Refresh now'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-7 gap-3">
          {kpiCards.map((kpi) => (
            <div key={kpi.label} className="bg-slate-900 border border-slate-800 p-3 rounded-lg shadow-sm">
              <span className="text-[9px] text-slate-500 font-bold uppercase">{kpi.label}</span>
              <div className={`text-xl font-mono font-bold mt-1 ${kpi.color}`}>
                {isLoading ? <span className="inline-block h-6 w-10 rounded bg-slate-800 animate-pulse" /> : (kpi.value ?? '—')}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ERROR BANNER */}
      {error && (
        <div className="shrink-0 flex flex-wrap items-center justify-between gap-3 bg-amber-950/30 border border-amber-700/50 text-amber-300 text-xs font-mono px-4 py-2 rounded-lg">
          <span>
            {stations.length ? 'Live refresh failed. Showing the last loaded data. ' : 'Station data could not be loaded. '}
            <span className="text-amber-500">{error}</span>
          </span>
          <span className="flex gap-2">
            <button type="button" onClick={loadAll} className="border border-amber-600/60 hover:bg-amber-900/40 px-3 py-1 rounded transition-colors">Retry</button>
            {source === 'api' && (
              <button type="button" onClick={() => switchSource('sim')} className="border border-amber-600/60 hover:bg-amber-900/40 px-3 py-1 rounded transition-colors">Use simulation</button>
            )}
          </span>
        </div>
      )}

      {/* FILTER BAR */}
      <div className="flex flex-wrap items-center gap-3 bg-slate-900/80 p-2 border border-slate-800 rounded-lg shrink-0">
        <div className="relative w-full sm:w-64">
          <input
            ref={searchRef}
            placeholder="Search station or zone…"
            aria-label="Search station or zone"
            value={filters.search}
            onChange={(e) => setFilters({ ...filters, search: e.target.value })}
            className="bg-slate-950 border border-slate-800 rounded px-3 py-1.5 pr-8 text-xs text-slate-200 w-full outline-none font-mono focus:border-red-500"
          />
          <kbd className="hidden sm:block absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-slate-600 border border-slate-800 rounded px-1">/</kbd>
        </div>
        <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by status">
          {statusPills.map((p) => (
            <button
              key={p.key}
              type="button"
              aria-pressed={filters.status === p.key}
              onClick={() => setFilters({ ...filters, status: p.key })}
              className={`px-2.5 py-1 rounded text-[11px] font-mono border transition-colors ${filters.status === p.key ? 'bg-slate-700 border-slate-500 text-white' : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'}`}
            >
              {p.label} <span className="text-slate-500">{p.count}</span>
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-xs font-mono text-slate-300">
          <input type="checkbox" checked={filters.active_only} onChange={(e) => setFilters({ ...filters, active_only: e.target.checked })} className="accent-red-500" />
          Active incidents only
        </label>
        <div className="ml-auto flex items-center gap-3 text-[10px] font-mono text-slate-500">
          <span>Showing {rows.length} of {stations.length}</span>
          {filtersActive && (
            <button type="button" onClick={() => setFilters({ search: '', status: 'ALL', active_only: false })} className="text-red-400 hover:text-red-300 underline">
              Clear filters
            </button>
          )}
        </div>
      </div>

      {/* MAIN GRID */}
      <div className="flex flex-col lg:flex-row flex-1 gap-4 lg:overflow-hidden min-h-0">

        {/* LEFT COLUMN: Table & panels */}
        <div className="lg:w-7/12 flex flex-col gap-4 lg:overflow-hidden min-h-0">

          {/* STATION TABLE */}
          <div className="flex-1 flex flex-col bg-slate-900 border border-slate-800 rounded-lg overflow-hidden min-h-[260px] max-h-[520px] lg:max-h-none">
            <div className="overflow-x-auto flex flex-col flex-1 min-h-0">
              <div className="min-w-[580px] flex flex-col flex-1 min-h-0">
                <div className="bg-slate-950 p-2 border-b border-slate-800 grid grid-cols-12 text-[10px] text-slate-400 shrink-0">
                  <SortHeader label="Station" k="name" sort={sort} onSort={onSort} className="col-span-4" />
                  <SortHeader label="Status" k="status" sort={sort} onSort={onSort} className="col-span-2" />
                  <SortHeader label="Active" k="incidents" sort={sort} onSort={onSort} className="col-span-2 justify-center" />
                  <SortHeader label="Engines free" k="engines" sort={sort} onSort={onSort} className="col-span-2 justify-center" />
                  <SortHeader label="Response" k="response" sort={sort} onSort={onSort} className="col-span-2 justify-end" />
                </div>
                <div className="overflow-y-auto custom-scrollbar flex-1">
                  {isLoading && stations.length === 0 && Array.from({ length: 6 }).map((_, i) => (
                    <div key={i} className="p-3 border-b border-slate-800/50"><div className="h-8 rounded bg-slate-800/60 animate-pulse" /></div>
                  ))}
                  {!isLoading && rows.length === 0 && (
                    <div className="p-8 text-center text-xs font-mono text-slate-500">
                      {stations.length === 0 ? 'No fire stations were returned by the server.' : 'No stations match these filters.'}
                    </div>
                  )}
                  {rows.map((st) => {
                    const status = getDerivedStatus(st);
                    const isSel = selectedId === st.entity_id;
                    const freeRatio = st.engines > 0 && st.engines_available != null ? st.engines_available / st.engines : null;
                    return (
                      <div
                        key={st.entity_id}
                        role="button"
                        tabIndex={0}
                        onClick={() => setSelectedId(st.entity_id)}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelectedId(st.entity_id); } }}
                        className={`grid grid-cols-12 p-3 border-b border-slate-800/50 text-xs items-center cursor-pointer hover:bg-slate-800/40 focus:outline-none focus-visible:bg-slate-800/60 ${isSel ? 'bg-slate-800/80 border-l-2 border-l-red-500' : ''}`}
                      >
                        <div className="col-span-4 flex flex-col gap-0.5 pr-2 min-w-0">
                          <span className="font-mono font-bold text-slate-200 truncate">{st.name}</span>
                          <span className="text-[10px] text-slate-500 font-mono truncate">
                            {st.zone || (hasCoords(st) ? `${st.latitude.toFixed(4)}, ${st.longitude.toFixed(4)}` : 'No coordinates')}
                          </span>
                        </div>
                        <div className="col-span-2">
                          <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold border uppercase ${StatusColors[status]}`}>{status}</span>
                        </div>
                        <div className="col-span-2 text-center font-mono font-bold text-slate-300">{st.active_incidents}</div>
                        <div className="col-span-2 flex flex-col items-center gap-1 font-mono text-slate-400">
                          {freeRatio != null ? (
                            <>
                              <span className="text-slate-300">{st.engines_available}/{st.engines}</span>
                              <span className="h-1 w-12 bg-slate-800 rounded overflow-hidden">
                                <span className="block h-full rounded" style={{ width: `${freeRatio * 100}%`, background: freeRatio === 0 ? '#ef4444' : freeRatio < 0.5 ? '#f97316' : '#22c55e' }} />
                              </span>
                            </>
                          ) : (st.engines > 0 ? `${st.engines} Total` : 'N/A')}
                        </div>
                        <div className="col-span-2 text-right font-mono text-slate-400">{st.avg_response_time ? `${st.avg_response_time.toFixed(1)} min` : '—'}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* READINESS & WORKLOAD / ACTIVITY */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 shrink-0 md:h-56">
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col">
              <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest border-b border-slate-800 pb-2 mb-3">Resource Readiness</h3>
              <div className="flex-1 flex flex-col justify-around gap-2 text-xs font-mono">
                <div className="flex justify-between items-center"><span className="text-slate-400">Total Fire Engines</span> <span className="font-bold text-slate-200">{kpis.engines ?? 'NOT AVAILABLE'}</span></div>
                {kpis.free != null && (
                  <div className="flex justify-between items-center"><span className="text-slate-400">Engines Available</span> <span className={`font-bold ${kpis.free / (kpis.engines || 1) < 0.5 ? 'text-orange-400' : 'text-emerald-400'}`}>{kpis.free}</span></div>
                )}
                <div className="flex justify-between items-center"><span className="text-slate-400">Tenders Dispatched</span> <span className="font-bold text-orange-400">{resources.fireTendersDispatched ?? 0}</span></div>
                <div className="flex justify-between items-center"><span className="text-slate-400">Rescue Units Disp.</span> <span className="font-bold text-orange-400">{resources.rescueUnitsDispatched ?? 0}</span></div>
                <div className="flex justify-between items-center"><span className="text-slate-400">Water Tankers Disp.</span> <span className="font-bold text-orange-400">{resources.waterTankersDispatched ?? 0}</span></div>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col min-h-[180px]">
              <div className="flex gap-4 border-b border-slate-800 pb-2 mb-3">
                {[['workload', 'Workload (Top 4)'], ['activity', 'Live Activity']].map(([k, label]) => (
                  <button
                    key={k}
                    type="button"
                    onClick={() => setLowerTab(k)}
                    aria-pressed={lowerTab === k}
                    className={`text-[10px] font-bold uppercase tracking-widest transition-colors ${lowerTab === k ? 'text-slate-100' : 'text-slate-600 hover:text-slate-400'}`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="flex-1 flex flex-col gap-2 overflow-y-auto custom-scrollbar pr-1">
                {lowerTab === 'workload' && (
                  <>
                    {topWorkload.map((w, i) => (
                      <button
                        type="button"
                        key={w.entity_id}
                        onClick={() => setSelectedId(w.entity_id)}
                        className="text-left text-xs font-mono bg-slate-950 p-2 rounded border border-slate-800 hover:border-slate-600 transition-colors"
                      >
                        <div className="flex justify-between items-center">
                          <span className="text-slate-300 truncate pr-2">{i + 1}. {w.name}</span>
                          <span className="text-red-400 font-bold whitespace-nowrap">{w.active_incidents} ACTIVE</span>
                        </div>
                        <div className="h-1 bg-slate-800 rounded mt-1.5 overflow-hidden">
                          <div className="h-full bg-red-500/80 rounded" style={{ width: `${(w.active_incidents / maxLoad) * 100}%` }} />
                        </div>
                      </button>
                    ))}
                    {topWorkload.length === 0 && <div className="text-xs text-slate-500 italic mt-2">No stations currently under load.</div>}
                  </>
                )}
                {lowerTab === 'activity' && (
                  <>
                    {activity.map((ev) => (
                      <button
                        type="button"
                        key={ev.id}
                        onClick={() => ev.station_id && setSelectedId(ev.station_id)}
                        className="text-left flex gap-2 items-start text-[11px] font-mono text-slate-300 hover:bg-slate-800/50 rounded p-1.5 transition-colors"
                      >
                        <span className={`mt-1 w-1.5 h-1.5 rounded-full shrink-0 ${EventDot[ev.kind] || 'bg-slate-500'}`} />
                        <span className="flex-1">{ev.text}</span>
                        <span className="text-slate-600 whitespace-nowrap">{timeAgo(ev.ts)}</span>
                      </button>
                    ))}
                    {activity.length === 0 && (
                      <div className="text-xs text-slate-500 italic mt-2">
                        {source === 'sim' ? 'Waiting for the next call…' : 'Activity feed is available in simulation mode.'}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Map */}
        <div className="lg:w-5/12 bg-slate-900 border border-slate-800 rounded-lg overflow-hidden relative min-h-[360px]">
          <div className="absolute top-3 left-3 z-[400] pointer-events-none">
            <span className="text-[10px] font-bold bg-slate-950/80 px-2 py-1 rounded border border-slate-700 text-slate-300 uppercase shadow-md">Mumbai Station Map</span>
          </div>
          <div className="absolute bottom-3 left-3 z-[400] bg-slate-950/85 border border-slate-700 rounded px-2.5 py-2 flex flex-col gap-1 text-[10px] font-mono text-slate-300">
            {Object.keys(MarkerColors).map((k) => (
              <span key={k} className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: MarkerColors[k] }} />{k}
              </span>
            ))}
            {liveIncidents.length > 0 && (
              <span className="flex items-center gap-2 border-t border-slate-700 pt-1 mt-0.5">
                <span className="w-2 h-2 rotate-45 bg-orange-400" />Active call ({liveIncidents.length})
              </span>
            )}
          </div>
          <MapContainer center={[19.0760, 72.8777]} zoom={11} style={{ height: '100%', width: '100%' }} zoomControl={false}>
           <TileLayer
  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
/>
            <ZoomControl position="bottomright" />
            <MapController stations={mapStations} selected={sel} />

            {/* Lines from the selected station to the calls it is handling */}
            {sel && liveIncidents.filter((i) => i.station_id === sel.entity_id && Number.isFinite(i.latitude)).map((i) => (
              <Polyline
                key={`line-${i.incident_number}`}
                positions={[[sel.latitude, sel.longitude], [i.latitude, i.longitude]]}
                pathOptions={{ color: IncidentHex[i.status], weight: 2, opacity: 0.7, dashArray: '4 6' }}
              />
            ))}

            {liveIncidents.filter((i) => Number.isFinite(i.latitude) && Number.isFinite(i.longitude)).map((i) => (
              <Marker
                key={i.incident_number}
                position={[i.latitude, i.longitude]}
                icon={incidentIcon(i.status, i.station_id === selectedId)}
                zIndexOffset={800}
              >
                <Popup className="fs-popup">
                  <div className="p-2 font-mono text-xs text-slate-800 min-w-[170px]">
                    <div className="font-bold mb-0.5">{pretty(i.incident_type)}</div>
                    <div className="text-slate-600">{i.location}</div>
                    <div className="mt-1 font-bold" style={{ color: IncidentHex[i.status] }}>{pretty(i.status)} · {i.elapsed_min} min</div>
                    <div className="text-slate-600">Level {i.call_level} · {i.engines_assigned} engine{i.engines_assigned > 1 ? 's' : ''} · {i.station_name}</div>
                  </div>
                </Popup>
              </Marker>
            ))}

            {mapStations.map((st) => {
              const status = getDerivedStatus(st);
              const isSel = selectedId === st.entity_id;
              return (
                <Marker
                  key={st.entity_id}
                  position={[st.latitude, st.longitude]}
                  icon={stationIcon(status, isSel)}
                  zIndexOffset={isSel ? 1000 : status === 'HIGH LOAD' ? 500 : 0}
                  eventHandlers={{ click: () => setSelectedId(st.entity_id) }}
                >
                  <Popup className="fs-popup">
                    <div className="p-2 font-mono text-xs text-slate-800 min-w-[150px]">
                      <div className="font-bold text-sm mb-1">{st.name}</div>
                      <div className="font-bold mb-1" style={{ color: MarkerColors[status] }}>{status}</div>
                      <div className="text-slate-600">Active incidents: {st.active_incidents}</div>
                      <div className="text-slate-600">Engines: {st.engines_available != null ? `${st.engines_available}/${st.engines} free` : (st.engines || 'N/A')}</div>
                    </div>
                  </Popup>
                </Marker>
              );
            })}
          </MapContainer>
        </div>
      </div>

      {/* NEW-CALL TOASTS */}
      <div className="fixed bottom-4 left-4 z-[1100] flex flex-col gap-2 max-w-sm" aria-live="polite">
        {toasts.map((t) => (
          <button
            type="button"
            key={t.id}
            onClick={() => { if (t.station_id) setSelectedId(t.station_id); setToasts((x) => x.filter((y) => y.id !== t.id)); }}
            className="fs-toast text-left bg-slate-900 border border-slate-700 border-l-4 border-l-red-500 rounded-md shadow-2xl px-3 py-2"
          >
            <div className="text-[9px] font-bold uppercase tracking-widest text-red-400">New call</div>
            <div className="text-xs font-mono text-slate-200 mt-0.5">{t.text}</div>
          </button>
        ))}
      </div>

      {/* DETAIL DRAWER */}
      {sel && (
        <button
          type="button"
          aria-label="Close station details"
          onClick={() => setSelectedId(null)}
          className="fixed inset-0 z-[900] bg-slate-950/50 sm:bg-transparent cursor-default"
        />
      )}
      <aside
        aria-hidden={!sel}
        className={`fixed top-16 bottom-0 right-0 w-full sm:w-[450px] bg-slate-900 border-l border-slate-700 shadow-2xl transition-transform duration-300 ease-in-out z-[1000] flex flex-col ${sel ? 'translate-x-0' : 'translate-x-full'}`}
      >
        {sel && (
          <>
            <div className="p-5 border-b border-slate-800 flex justify-between items-start bg-slate-950/50 shrink-0">
              <div className="min-w-0">
                <div className="text-[10px] font-mono text-slate-400 mb-1">FIRE STATION{sel.zone ? ` · ${sel.zone.toUpperCase()}` : ''}</div>
                <h2 className="text-lg font-bold text-slate-100 uppercase truncate">{sel.name}</h2>
                <div className="flex gap-2 mt-2">
                  <span className={`text-[9px] px-2 py-0.5 rounded font-bold border uppercase tracking-widest ${StatusColors[selStatus]}`}>
                    DERIVED STATUS: {selStatus}
                  </span>
                </div>
              </div>
              <button type="button" aria-label="Close" onClick={() => setSelectedId(null)} className="text-slate-500 hover:text-white bg-slate-900 p-1.5 rounded border border-slate-800">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
              </button>
            </div>

            <div className="p-5 flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-6">
              {detail.error && (
                <div className="text-[11px] font-mono text-amber-300 bg-amber-950/30 border border-amber-700/50 rounded px-3 py-2">{detail.error}</div>
              )}

              <div>
                <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest border-b border-slate-800 pb-2 mb-3">Station Info</h3>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div><span className="text-slate-500 block text-[9px] uppercase font-bold">Coordinates</span> <span className="text-slate-200 font-mono">{hasCoords(sel) ? `${sel.latitude.toFixed(5)}, ${sel.longitude.toFixed(5)}` : '—'}</span></div>
                  <div><span className="text-slate-500 block text-[9px] uppercase font-bold">Layer Group</span> <span className="text-slate-200 font-mono">{sel.layer_group || '—'}</span></div>
                  {sel.crew != null && <div><span className="text-slate-500 block text-[9px] uppercase font-bold">Crew on duty</span> <span className="text-slate-200 font-mono">{sel.crew}</span></div>}
                  {sel.ladders != null && <div><span className="text-slate-500 block text-[9px] uppercase font-bold">Ladder units</span> <span className="text-slate-200 font-mono">{sel.ladders}</span></div>}
                  <div><span className="text-slate-500 block text-[9px] uppercase font-bold">Registry ID</span> <span className="text-slate-400 font-mono truncate block" title={sel.entity_id}>{String(sel.entity_id).split('-')[0]}...</span></div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-950 p-3 rounded border border-slate-800">
                  <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Resource Info</h3>
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-500">Fire Engines</span>
                    <span className="font-bold text-slate-200">{sel.engines > 0 ? sel.engines : 'N/A'}</span>
                  </div>
                  {sel.engines_available != null && sel.engines > 0 && (
                    <div className="mt-2">
                      <div className="flex justify-between text-[10px] font-mono text-slate-500 mb-1">
                        <span>Available</span><span>{sel.engines_available}/{sel.engines}</span>
                      </div>
                      <div className="h-1.5 bg-slate-800 rounded overflow-hidden">
                        <div className="h-full rounded transition-all" style={{ width: `${(sel.engines_available / sel.engines) * 100}%`, background: sel.engines_available === 0 ? '#ef4444' : sel.engines_available < sel.engines / 2 ? '#f97316' : '#22c55e' }} />
                      </div>
                    </div>
                  )}
                </div>
                <div className="bg-slate-950 p-3 rounded border border-slate-800">
                  <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Performance</h3>
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-500">Avg Response</span>
                    <span className="font-bold text-slate-200">{sel.avg_response_time ? `${sel.avg_response_time.toFixed(1)} min` : '—'}</span>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest border-b border-slate-800 pb-2 mb-3">Incident Load Summary</h3>
                <div className="grid grid-cols-2 gap-y-2 text-xs font-mono">
                  <div className="flex justify-between pr-4"><span className="text-slate-500">Active Incidents</span> <span className="text-red-400 font-bold">{sel.active_incidents}</span></div>
                  <div className="flex justify-between pr-4"><span className="text-slate-500">Today's Total</span> <span className="text-slate-200">{sel.today_incidents ?? '—'}</span></div>
                  <div className="flex justify-between pr-4"><span className="text-slate-500">Dispatched</span> <span className="text-indigo-400">{sel.dispatched_incidents ?? '—'}</span></div>
                  <div className="flex justify-between pr-4"><span className="text-slate-500">En Route</span> <span className="text-purple-400">{sel.en_route_incidents ?? '—'}</span></div>
                  <div className="flex justify-between pr-4"><span className="text-slate-500">On Scene</span> <span className="text-orange-400">{sel.on_scene_incidents ?? '—'}</span></div>
                  <div className="flex justify-between pr-4"><span className="text-slate-500">Contained</span> <span className="text-yellow-400">{sel.contained_incidents ?? '—'}</span></div>
                </div>
              </div>

              <div className="flex-1 flex flex-col">
                <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest border-b border-slate-800 pb-2 mb-3">Assigned Incidents</h3>
                <div className="flex flex-col gap-2">
                  {detail.loading ? (
                    [0, 1].map((i) => <div key={i} className="h-16 rounded bg-slate-800/60 animate-pulse" />)
                  ) : detail.incidents.length === 0 ? (
                    <div className="text-xs text-slate-500 font-mono italic p-4 text-center border border-slate-800 rounded bg-slate-950/50">No active incidents assigned. Station is ready for the next call.</div>
                  ) : (
                    detail.incidents.map((inc, i) => (
                      <div key={inc.incident_number || i} className="bg-slate-950 border border-slate-800 p-3 rounded flex flex-col gap-1">
                        <div className="flex justify-between items-start gap-2">
                          <span className="font-mono text-xs font-bold text-slate-200">{inc.incident_number}</span>
                          <span className={`text-[9px] font-bold uppercase tracking-widest ${IncidentText[inc.status] || 'text-slate-400'}`}>{pretty(inc.status)}</span>
                        </div>
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-xs text-slate-300 capitalize">{pretty(inc.incident_type).toLowerCase()}</span>
                          {inc.call_level != null && (
                            <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${inc.call_level >= 2 ? 'text-red-400 border-red-500/40 bg-red-500/10' : 'text-slate-400 border-slate-700'}`}>Level {inc.call_level}</span>
                          )}
                          {inc.engines_assigned != null && <span className="text-[10px] font-mono text-slate-500">{inc.engines_assigned} engine{inc.engines_assigned > 1 ? 's' : ''}</span>}
                          {inc.elapsed_min != null && <span className="text-[10px] font-mono text-slate-500 ml-auto">{inc.elapsed_min} min elapsed</span>}
                        </div>
                        <span className="text-[10px] font-mono text-slate-500 truncate">{[inc.location, inc.address].filter(Boolean).join(' • ')}</span>
                        {stepIdx(inc.status) >= 0 && (
                          <div className="flex gap-1 mt-1.5" aria-label={`Progress: ${pretty(inc.status)}`}>
                            {STEPS.map((s, n) => (
                              <div key={s} className="h-1 flex-1 rounded" style={{ background: n <= stepIdx(inc.status) ? IncidentHex[inc.status] : '#1e293b' }} title={pretty(s)} />
                            ))}
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}