// frontend/src/pages/Departments/Fire/components/FireOperationsMap.jsx
import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Popup, ZoomControl, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const SEVERITY = {
  CRITICAL: { color: '#ef4444', label: 'Critical', rank: 0 },
  HIGH: { color: '#f97316', label: 'High', rank: 1 },
  MODERATE: { color: '#eab308', label: 'Moderate', rank: 2 },
  LOW: { color: '#22c55e', label: 'Low', rank: 3 },
};
const FALLBACK_COLOR = '#3b82f6';

const BASEMAPS = {
  street: {
    label: 'OpenStreetMap',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
};  ``

const MUMBAI = [19.076, 72.8777];

// Popup + marker styling lives in one injected stylesheet so Leaflet's own
// white popup chrome can be themed (Tailwind classes on <Popup> don't reach it).
const MAP_CSS = `
.fire-map .leaflet-popup-content-wrapper {
  background: #0f172a; color: #e2e8f0; border: 1px solid #334155;
  border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,.5); padding: 0;
}
.fire-map .leaflet-popup-content { margin: 0; min-width: 190px; }
.fire-map .leaflet-popup-tip { background: #0f172a; border: 1px solid #334155; }
.fire-map .leaflet-popup-close-button { color: #94a3b8 !important; top: 6px; right: 6px; }
.fire-map .leaflet-popup-close-button:hover { color: #f1f5f9 !important; }
.fire-map .leaflet-container { background: #0b1120; font-family: inherit; }
.fire-map .leaflet-control-zoom a {
  background: #0f172a; color: #e2e8f0; border-color: #334155;
}
.fire-map .leaflet-control-zoom a:hover { background: #1e293b; }
.fire-map .leaflet-control-attribution {
  background: rgba(15,23,42,.8); color: #94a3b8; font-size: 10px;
}
.fire-map .leaflet-control-attribution a { color: #94a3b8; }
.fire-marker { background: transparent; border: 0; }
.fire-marker .dot {
  position: relative; border-radius: 9999px; border: 2px solid #fff;
  transition: transform .15s ease;
}
.fire-marker:hover .dot, .fire-marker:focus-visible .dot { transform: scale(1.25); }
.fire-marker:focus-visible { outline: 2px solid #fff; outline-offset: 3px; border-radius: 9999px; }
.fire-marker .pulse {
  position: absolute; inset: -2px; border-radius: 9999px;
  animation: fire-pulse 1.8s ease-out infinite;
}
.fire-marker .ring {
  position: absolute; inset: -9px; border-radius: 9999px;
  border: 2px solid #fff; opacity: .9;
}
@keyframes fire-pulse {
  0% { transform: scale(1); opacity: .7; }
  100% { transform: scale(3); opacity: 0; }
}
@media (prefers-reduced-motion: reduce) {
  .fire-marker .pulse { animation: none; opacity: 0; }
  .fire-marker .dot { transition: none; }
}
`;

const hasCoords = (i) =>
  Number.isFinite(Number(i?.latitude)) && Number.isFinite(Number(i?.longitude));

const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

const formatType = (t) => (t ? String(t).replace(/_/g, ' ') : 'Unknown type');

const createCustomIcon = (severity, isSelected) => {
  const color = SEVERITY[severity]?.color || FALLBACK_COLOR;
  const size = severity === 'CRITICAL' ? 20 : 16;
  const pulse = severity === 'CRITICAL' ? `<span class="pulse" style="background:${color}"></span>` : '';
  const ring = isSelected ? '<span class="ring"></span>' : '';
  return L.divIcon({
    className: 'fire-marker',
    html: `<div class="dot" style="background:${color};width:${size}px;height:${size}px;box-shadow:0 0 12px ${color}">${pulse}${ring}</div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2 - 2],
  });
};

/**
 * Moves the camera. Fits all incidents only when the SET of incidents changes
 * (not on every refresh), so live polling never yanks the view away from
 * what the operator is looking at.
 */
const MapController = ({ incidents, selectedIncident, fitSignal }) => {
  const map = useMap();
  const lastFitKey = useRef('');

  const fitAll = useCallback(() => {
    if (incidents.length === 0) return;
    const bounds = L.latLngBounds(incidents.map((i) => [Number(i.latitude), Number(i.longitude)]));
    map.fitBounds(bounds, {
      padding: [60, 60],
      maxZoom: 14,
      animate: !prefersReducedMotion(),
    });
  }, [incidents, map]);

  // Fly to the selected incident.
  useEffect(() => {
    if (selectedIncident && hasCoords(selectedIncident)) {
      map.flyTo([Number(selectedIncident.latitude), Number(selectedIncident.longitude)], 16, {
        animate: !prefersReducedMotion(),
        duration: 1.2,
      });
    }
  }, [selectedIncident?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fit when the set of visible incidents changes and nothing is selected.
  useEffect(() => {
    const key = incidents.map((i) => i.id).sort().join(',');
    if (key !== lastFitKey.current) {
      lastFitKey.current = key;
      if (!selectedIncident) fitAll();
    }
  }, [incidents, selectedIncident, fitAll]);

  // Explicit "Show all" button.
  useEffect(() => {
    if (fitSignal > 0) fitAll();
  }, [fitSignal]); // eslint-disable-line react-hooks/exhaustive-deps

  return null;
};

const MapButton = ({ active, children, ...props }) => (
  <button
    type="button"
    aria-pressed={active}
    className={`px-2.5 py-1.5 text-xs font-medium rounded-md border transition-colors
      focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400
      ${active
        ? 'bg-slate-700 border-slate-500 text-white'
        : 'bg-slate-900/90 border-slate-700 text-slate-300 hover:bg-slate-800'}`}
    {...props}
  >
    {children}
  </button>
);

export default function FireOperationsMap({ incidents = [], selectedIncident, onSelectIncident }) {
  const [basemap, setBasemap] = useState('dark');
  const [hidden, setHidden] = useState(() => new Set()); // severities toggled off
  const [fitSignal, setFitSignal] = useState(0);

  const mappable = useMemo(() => incidents.filter(hasCoords), [incidents]);
  const skipped = incidents.length - mappable.length;

  const counts = useMemo(() => {
    const c = { CRITICAL: 0, HIGH: 0, MODERATE: 0, LOW: 0 };
    mappable.forEach((i) => {
      if (c[i.severity] !== undefined) c[i.severity] += 1;
    });
    return c;
  }, [mappable]);

  const visible = useMemo(
    () => mappable.filter((i) => !hidden.has(i.severity)),
    [mappable, hidden]
  );

  // Draw critical incidents last so they are never buried under lower severities.
  const ordered = useMemo(
    () =>
      [...visible].sort(
        (a, b) => (SEVERITY[b.severity]?.rank ?? 9) - (SEVERITY[a.severity]?.rank ?? 9)
      ),
    [visible]
  );

  const toggleSeverity = (sev) =>
    setHidden((prev) => {
      const next = new Set(prev);
      next.has(sev) ? next.delete(sev) : next.add(sev);
      return next;
    });

  const tiles = BASEMAPS[basemap];

  return (
    <div className="fire-map w-full h-full rounded-lg overflow-hidden border border-slate-700 relative bg-slate-900 z-0">
      <style>{MAP_CSS}</style>

      <MapContainer
        center={MUMBAI}
        zoom={11}
        style={{ height: '100%', width: '100%' }}
        zoomControl={false}
        keyboard
      >
        <TileLayer
  key={basemap}
  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
/>
        <ZoomControl position="bottomright" />
        <MapController
          incidents={visible}
          selectedIncident={selectedIncident}
          fitSignal={fitSignal}
        />

        {ordered.map((incident) => {
          const isSelected = selectedIncident?.id === incident.id;
          const color = SEVERITY[incident.severity]?.color || FALLBACK_COLOR;
          return (
            <Marker
              key={`${incident.id}-${incident.severity}-${isSelected}`}
              position={[Number(incident.latitude), Number(incident.longitude)]}
              icon={createCustomIcon(incident.severity, isSelected)}
              zIndexOffset={isSelected ? 1000 : 0}
              title={`${incident.incident_number} – ${formatType(incident.incident_type)}`}
              eventHandlers={{ click: () => onSelectIncident?.(incident) }}
            >
              <Popup closeButton autoPan>
                <div className="p-3 font-mono text-xs">
                  <div className="flex items-center gap-2">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className="font-bold text-sm text-slate-100">
                      {incident.incident_number}
                    </span>
                  </div>
                  <div className="font-semibold mt-1.5 capitalize" style={{ color }}>
                    {formatType(incident.incident_type).toLowerCase()}
                  </div>
                  <div className="text-slate-400 mt-1">
                    {[incident.status, incident.location].filter(Boolean).join(' • ')}
                  </div>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>

      {/* Top-left: legend that doubles as a severity filter */}
      <div
        className="absolute top-3 left-3 z-[1000] rounded-lg border border-slate-700 bg-slate-900/90 backdrop-blur p-2 shadow-lg"
        role="group"
        aria-label="Filter incidents by severity"
      >
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(SEVERITY).map(([key, { color, label }]) => {
            const off = hidden.has(key);
            return (
              <button
                key={key}
                type="button"
                aria-pressed={!off}
                onClick={() => toggleSeverity(key)}
                className={`flex items-center gap-1.5 px-2 py-1 rounded-md border text-xs transition
                  focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400
                  ${off
                    ? 'border-slate-800 text-slate-500 bg-transparent'
                    : 'border-slate-600 text-slate-200 bg-slate-800/70 hover:bg-slate-800'}`}
              >
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: off ? '#475569' : color }}
                />
                {label}
                <span className="tabular-nums text-slate-400">{counts[key]}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Top-right: view controls */}
      <div className="absolute top-3 right-3 z-[1000] flex gap-1.5">
        <MapButton onClick={() => setFitSignal((n) => n + 1)} title="Zoom to show all visible incidents">
          Show all
        </MapButton>
        {Object.entries(BASEMAPS).map(([key, { label }]) => (
          <MapButton key={key} active={basemap === key} onClick={() => setBasemap(key)}>
            {label}
          </MapButton>
        ))}
      </div>

      {/* Empty / filtered-out state */}
      {visible.length === 0 && (
        <div className="absolute inset-0 z-[900] flex items-center justify-center pointer-events-none">
          <div className="rounded-lg border border-slate-700 bg-slate-900/90 px-4 py-3 text-center max-w-xs">
            <p className="text-sm font-medium text-slate-100">
              {mappable.length === 0 ? 'No incidents to show' : 'All incidents are filtered out'}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              {mappable.length === 0
                ? 'New incidents will appear here as they are reported.'
                : 'Turn a severity back on in the top-left to see them.'}
            </p>
          </div>
        </div>
      )}

      {/* Data-quality notice */}
      {skipped > 0 && (
        <div
          className="absolute bottom-3 left-3 z-[1000] rounded-md border border-amber-700/60 bg-slate-900/90 px-2.5 py-1.5 text-xs text-amber-300"
          role="status"
        >
          {skipped} {skipped === 1 ? 'incident has' : 'incidents have'} no location and can't be shown on the map
        </div>
      )}
    </div>
  );
}