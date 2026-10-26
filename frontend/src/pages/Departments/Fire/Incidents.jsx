// frontend/src/pages/Departments/Fire/components/Incidents.jsx
import React, { useEffect, useMemo, useRef, useState } from 'react';
import FireOperationsMap from './components/FireOperationsMap';

const SEVERITY = {
  CRITICAL: { color: '#ef4444', label: 'Critical', rank: 0 },
  HIGH: { color: '#f97316', label: 'High', rank: 1 },
  MODERATE: { color: '#eab308', label: 'Moderate', rank: 2 },
  LOW: { color: '#22c55e', label: 'Low', rank: 3 },
};

const formatType = (t) => (t ? String(t).replace(/_/g, ' ').toLowerCase() : 'unknown type');
const formatStatus = (s) => (s ? String(s).replace(/_/g, ' ').toLowerCase() : 'unknown');

const timeAgo = (value) => {
  if (!value) return '';
  const mins = Math.round((Date.now() - new Date(value).getTime()) / 60000);
  if (Number.isNaN(mins)) return '';
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs} h ago`;
  return `${Math.round(hrs / 24)} d ago`;
};

const SORTS = {
  severity: 'Most severe first',
  newest: 'Newest first',
};

function IncidentRow({ incident, selected, onSelect }) {
  const ref = useRef(null);
  const sev = SEVERITY[incident.severity];
  const color = sev?.color || '#3b82f6';

  // Keep the row visible when the selection came from a map click.
  useEffect(() => {
    if (selected) ref.current?.scrollIntoView({ block: 'nearest' });
  }, [selected]);

  return (
    <li>
      <button
        ref={ref}
        type="button"
        onClick={() => onSelect(incident)}
        aria-current={selected ? 'true' : undefined}
        className={`w-full text-left flex gap-3 px-3 py-3 border-b border-slate-800 transition-colors
          focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-sky-400
          ${selected ? 'bg-slate-800' : 'hover:bg-slate-800/60'}`}
      >
        <span
          className="w-1 rounded-full shrink-0"
          style={{ backgroundColor: color }}
          aria-hidden="true"
        />
        <span className="min-w-0 flex-1">
          <span className="flex items-baseline justify-between gap-2">
            <span className="font-mono text-sm font-semibold text-slate-100 truncate">
              {incident.incident_number}
            </span>
            <span className="text-xs text-slate-500 shrink-0">
              {timeAgo(incident.created_at)}
            </span>
          </span>
          <span className="block text-sm capitalize mt-0.5" style={{ color }}>
            {formatType(incident.incident_type)}
            <span className="sr-only">, {sev?.label || 'unknown'} severity</span>
          </span>
          <span className="flex items-center gap-2 mt-1 text-xs text-slate-400">
            <span className="rounded border border-slate-700 px-1.5 py-0.5 capitalize">
              {formatStatus(incident.status)}
            </span>
            <span className="truncate">{incident.location}</span>
          </span>
        </span>
      </button>
    </li>
  );
}

// Loads incidents itself unless the parent passes an `incidents` array.
// If your app uses an axios instance or auth headers, swap them in below.
function useIncidentsFeed(endpoint, enabled, pollMs) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState(null);

  const load = React.useCallback(async () => {
    if (!enabled) return;
    setLoading(true);
    try {
      const res = await fetch(endpoint, { credentials: 'include' });
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      const json = await res.json();
      const rows = Array.isArray(json) ? json : json.data || json.incidents || [];
      setData(rows);
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, [endpoint, enabled]);

  useEffect(() => {
    load();
    if (!enabled || !pollMs) return undefined;
    const t = setInterval(load, pollMs);
    return () => clearInterval(t);
  }, [load, enabled, pollMs]);

  return { data, loading, error, reload: load };
}

export default function Incidents({
  incidents: incidentsProp,
  loading: loadingProp,
  error: errorProp = null,
  onRefresh: onRefreshProp,
  endpoint = '/api/fire/incidents?active_only=true',
  pollMs = 30000,
}) {
  const external = Array.isArray(incidentsProp);
  const feed = useIncidentsFeed(endpoint, !external, pollMs);
  const incidents = external ? incidentsProp : feed.data;
  const loading = external ? Boolean(loadingProp) : feed.loading;
  const error = external ? errorProp : feed.error;
  const onRefresh = external ? onRefreshProp : feed.reload;

  const [selected, setSelected] = useState(null);
  const [query, setQuery] = useState('');
  const [status, setStatus] = useState('ALL');
  const [sort, setSort] = useState('severity');

  const statuses = useMemo(
    () => [...new Set(incidents.map((i) => i.status).filter(Boolean))],
    [incidents]
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = incidents.filter((i) => {
      if (status !== 'ALL' && i.status !== status) return false;
      if (!q) return true;
      return [i.incident_number, i.incident_type, i.location, i.status]
        .filter(Boolean)
        .some((v) => String(v).toLowerCase().replace(/_/g, ' ').includes(q));
    });
    return list.sort((a, b) => {
      if (sort === 'newest') {
        return new Date(b.created_at || 0) - new Date(a.created_at || 0);
      }
      const r = (SEVERITY[a.severity]?.rank ?? 9) - (SEVERITY[b.severity]?.rank ?? 9);
      return r || new Date(b.created_at || 0) - new Date(a.created_at || 0);
    });
  }, [incidents, query, status, sort]);

  // Drop the selection if filters hide the selected incident.
  useEffect(() => {
    if (selected && !filtered.some((i) => i.id === selected.id)) setSelected(null);
  }, [filtered, selected]);

  const critical = incidents.filter((i) => i.severity === 'CRITICAL').length;
  const filtersActive = query || status !== 'ALL';

  const clearFilters = () => {
    setQuery('');
    setStatus('ALL');
  };

  return (
    <div className="flex flex-col h-full min-h-[600px] bg-slate-950 text-slate-200">
      <header className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-slate-800">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">Incidents</h1>
          <p className="text-xs text-slate-400" aria-live="polite">
            {incidents.length} active
            {critical > 0 && (
              <span className="text-red-400 font-medium"> · {critical} critical</span>
            )}
          </p>
        </div>
        {onRefresh && (
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className="px-3 py-1.5 text-sm rounded-md border border-slate-700 bg-slate-900 hover:bg-slate-800
              disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        )}
      </header>

      <div className="flex flex-1 min-h-0 flex-col-reverse lg:flex-row">
        {/* List pane */}
        <section
          className="lg:w-[380px] lg:shrink-0 flex flex-col min-h-0 h-80 lg:h-auto border-t lg:border-t-0 lg:border-r border-slate-800"
          aria-label="Incident list"
        >
          <div className="p-3 space-y-2 border-b border-slate-800">
            <label className="sr-only" htmlFor="incident-search">Search incidents</label>
            <input
              id="incident-search"
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by number, type or location"
              className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm
                placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-400"
            />
            <div className="flex gap-2">
              <label className="sr-only" htmlFor="incident-status">Filter by status</label>
              <select
                id="incident-status"
                value={status}
                onChange={(e) => setStatus(e.target.value)}
                className="flex-1 rounded-md bg-slate-900 border border-slate-700 px-2 py-1.5 text-sm capitalize
                  focus:outline-none focus:ring-2 focus:ring-sky-400"
              >
                <option value="ALL">All statuses</option>
                {statuses.map((s) => (
                  <option key={s} value={s}>{formatStatus(s)}</option>
                ))}
              </select>
              <label className="sr-only" htmlFor="incident-sort">Sort incidents</label>
              <select
                id="incident-sort"
                value={sort}
                onChange={(e) => setSort(e.target.value)}
                className="flex-1 rounded-md bg-slate-900 border border-slate-700 px-2 py-1.5 text-sm
                  focus:outline-none focus:ring-2 focus:ring-sky-400"
              >
                {Object.entries(SORTS).map(([k, v]) => (
                  <option key={k} value={k}>{v}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto" aria-busy={loading}>
            {error ? (
              <div className="p-4 text-sm" role="alert">
                <p className="font-medium text-red-400">Couldn't load incidents</p>
                <p className="text-slate-400 mt-1">{String(error.message || error)}</p>
                {onRefresh && (
                  <button
                    type="button"
                    onClick={onRefresh}
                    className="mt-3 px-3 py-1.5 rounded-md border border-slate-700 hover:bg-slate-800"
                  >
                    Try again
                  </button>
                )}
              </div>
            ) : loading && incidents.length === 0 ? (
              <ul className="animate-pulse" aria-label="Loading incidents">
                {[0, 1, 2, 3, 4].map((n) => (
                  <li key={n} className="px-3 py-3 border-b border-slate-800">
                    <div className="h-3 w-28 bg-slate-800 rounded" />
                    <div className="h-3 w-40 bg-slate-800 rounded mt-2" />
                    <div className="h-3 w-52 bg-slate-800 rounded mt-2" />
                  </li>
                ))}
              </ul>
            ) : filtered.length === 0 ? (
              <div className="p-6 text-center text-sm">
                <p className="font-medium text-slate-100">
                  {filtersActive ? 'No incidents match your filters' : 'No active incidents'}
                </p>
                <p className="text-slate-400 mt-1">
                  {filtersActive
                    ? 'Try a different search or status.'
                    : 'New incidents will appear here as they are reported.'}
                </p>
                {filtersActive && (
                  <button
                    type="button"
                    onClick={clearFilters}
                    className="mt-3 px-3 py-1.5 rounded-md border border-slate-700 hover:bg-slate-800"
                  >
                    Clear filters
                  </button>
                )}
              </div>
            ) : (
              <ul>
                {filtered.map((incident) => (
                  <IncidentRow
                    key={incident.id}
                    incident={incident}
                    selected={selected?.id === incident.id}
                    onSelect={setSelected}
                  />
                ))}
              </ul>
            )}
          </div>

          <p className="px-3 py-2 text-xs text-slate-500 border-t border-slate-800">
            Showing {filtered.length} of {incidents.length}
          </p>
        </section>

        {/* Map pane */}
        <section className="flex-1 min-h-[320px] p-3" aria-label="Incident map">
          <FireOperationsMap
            incidents={filtered}
            selectedIncident={selected}
            onSelectIncident={setSelected}
          />
        </section>
      </div>
    </div>
  );
}