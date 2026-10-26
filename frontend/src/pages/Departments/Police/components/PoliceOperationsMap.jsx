import React, { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import L from "leaflet";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  LayersControl,
  LayerGroup,
  ZoomControl,
  useMap,
  useMapEvents,
} from "react-leaflet";

// NOTE: adjust this path to wherever policeOverviewData.js lives in your project.
import {
  formatDateTime,
  getStationMeta,
  norm,
  parseDepartments,
  titleCase,
  toNum,
} from "../data/Policeoverviewdata";

// ============================================================
// SVG ICONS (inside markers)
// ============================================================

const stationSvg = `<svg width="14" height="14" fill="#60a5fa" viewBox="0 0 24 24"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/></svg>`;
const incidentSvg = `<svg width="14" height="14" fill="#f87171" viewBox="0 0 24 24"><path d="M12 2L2 20h20L12 2zm0 4.5L18.5 18h-13L12 6.5zM11 10v4h2v-4h-2zm0 5v2h2v-2h-2z"/></svg>`;
const crimeSvg = `<svg width="12" height="12" fill="#c084fc" viewBox="0 0 24 24"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-1 5h2v4h-2zm0 6h2v2h-2z"/></svg>`;

const KIND_STYLE = {
  station: { color: "#3b82f6", svg: stationSvg, dot: 24 },
  incident: { color: "#ef4444", svg: incidentSvg, dot: 28 },
  crime: { color: "#a855f7", svg: crimeSvg, dot: 20 },
};

// ============================================================
// MARKER ICONS (cached; pulse only for critical/high incidents)
// ============================================================

const iconCache = new Map();

const getIcon = (kind, { pulse = false, selected = false } = {}) => {
  const key = `${kind}-${pulse}-${selected}`;
  if (iconCache.has(key)) return iconCache.get(key);

  const { color, svg, dot } = KIND_STYLE[kind];
  const dotSize = selected ? dot + 8 : dot;
  const ring = dotSize + 12;

  const icon = L.divIcon({
    className: "custom-police-map-marker",
    html: `
      <div style="position:relative;width:${ring}px;height:${ring}px;display:flex;align-items:center;justify-content:center;">
        ${
          pulse
            ? `<div style="position:absolute;width:${ring}px;height:${ring}px;background:${color};opacity:0.3;border-radius:50%;animation:police-map-ping 1.6s cubic-bezier(0,0,0.2,1) infinite;"></div>`
            : ""
        }
        <div style="width:${dotSize}px;height:${dotSize}px;background:#0f172a;border:2px solid ${color};border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 0 ${
          selected ? "14px" : "5px"
        } ${color};z-index:100;">${svg}</div>
      </div>`,
    iconSize: [ring, ring],
    iconAnchor: [ring / 2, ring / 2],
    popupAnchor: [0, -(ring / 2)],
  });

  iconCache.set(key, icon);
  return icon;
};

// ============================================================
// POPUP
// ============================================================

const SEVERITY_TEXT = {
  CRITICAL: "text-red-400",
  HIGH: "text-orange-400",
  MEDIUM: "text-amber-400",
  LOW: "text-emerald-400",
};

const PopupCard = ({ accent, kicker, title, rows, children }) => (
  <div className="w-72 bg-slate-900 text-slate-100 p-3 rounded-lg font-sans">
    <div className={`text-[9px] font-mono uppercase tracking-widest mb-0.5 ${accent}`}>
      {kicker}
    </div>
    <h3 className="font-bold text-sm text-slate-100 mb-2 pb-2 border-b border-slate-700 leading-tight">
      {title}
    </h3>

    <div className="grid grid-cols-2 gap-1.5 text-[10px]">
      {rows
        .filter((r) => r.value !== null && r.value !== undefined && r.value !== "")
        .map((row) => (
          <div
            key={row.label}
            className={`bg-slate-800 p-1.5 rounded min-w-0 ${row.wide ? "col-span-2" : ""}`}
          >
            <div className="text-slate-400">{row.label}</div>
            <div
              className={`truncate ${row.mono ? "font-mono" : ""} ${row.className || "text-slate-100"}`}
              title={String(row.value)}
            >
              {row.value}
            </div>
          </div>
        ))}
    </div>

    {children}
  </div>
);

const humanizeKey = (key) =>
  titleCase(String(key).replace(/([a-z])([A-Z])/g, "$1 $2"));

// ============================================================
// MAP HELPERS (inside MapContainer)
// ============================================================

function FlyToTarget({ target }) {
  const map = useMap();

  React.useEffect(() => {
    if (!target) return;
    map.flyTo([target.lat, target.lng], Math.max(map.getZoom(), 14), {
      duration: 0.8,
    });
  }, [target, map]);

  return null;
}

function ClearSelectionOnMapClick({ onClear }) {
  useMapEvents({ click: onClear });
  return null;
}

// ============================================================
// MAIN COMPONENT
// All data comes from props, so the map and the dashboard share one
// source of truth. No counters are rendered anywhere on the map.
// ============================================================

export default function PoliceOperationsMap({
  stations = [],
  incidents = [],
  crimes = [],
  error = null,
  highlightedIncidentId = null,
  // Set true to navigate immediately on marker click instead of showing
  // the popup first (the popup always has a "View Incident" button).
  navigateOnMarkerClick = false,
}) {
  const navigate = useNavigate();
  const [selectedId, setSelectedId] = useState(null);
  const [flyTarget, setFlyTarget] = useState(null);

  const openIncident = useCallback(
    (id) => navigate(`/police/incidents?id=${encodeURIComponent(id)}`),
    [navigate],
  );

  const handleIncidentClick = (incident, lat, lng) => {
    setSelectedId(incident.incident_id);
    setFlyTarget({ lat, lng, key: `${incident.incident_id}-${Date.now()}` });
    if (navigateOnMarkerClick) openIncident(incident.incident_id);
  };

  const clearSelection = useCallback(() => setSelectedId(null), []);

  return (
    <div className="relative w-full h-full bg-slate-950 overflow-hidden">
      <style>{`@keyframes police-map-ping{75%,100%{transform:scale(1.9);opacity:0}}`}</style>

      <MapContainer
        center={[19.076, 72.8777]}
        zoom={12}
        className="absolute inset-0 z-0"
        zoomControl={false}
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          className="dark-map-tiles"
        />

        <ZoomControl position="bottomright" />
        <FlyToTarget target={flyTarget} />
        <ClearSelectionOnMapClick onClear={clearSelection} />

        <LayersControl position="topright">
          {/* ---------------- POLICE STATIONS ---------------- */}
          <LayersControl.Overlay checked name="Police Stations">
            <LayerGroup>
              {stations.map((station) => {
                const lat = Number(station.latitude);
                const lng = Number(station.longitude);
                if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;

                const meta = getStationMeta(station);
                const available = toNum(meta.patrolsAvailable);
                const total = toNum(meta.patrolsTotal ?? meta.totalPatrols);
                const handled = new Set([
                  "patrolsAvailable",
                  "patrolsTotal",
                  "totalPatrols",
                  "status",
                ]);
                const extraRows = Object.entries(meta)
                  .filter(
                    ([k, v]) =>
                      !handled.has(k) && v !== null && typeof v !== "object" && v !== "",
                  )
                  .slice(0, 4)
                  .map(([k, v]) => ({ label: humanizeKey(k), value: String(v) }));

                return (
                  <Marker
                    key={station.entity_id || `station-${lat}-${lng}`}
                    position={[lat, lng]}
                    icon={getIcon("station")}
                    zIndexOffset={500}
                  >
                    <Popup className="enterprise-popup">
                      <PopupCard
                        accent="text-blue-400"
                        kicker="Police Station"
                        title={station.name || "Police Station"}
                        rows={[
                          { label: "Station ID", value: station.entity_id, mono: true },
                          {
                            label: "Status",
                            value: titleCase(meta.status || "Operational"),
                            className: "text-emerald-400",
                          },
                          {
                            label: "Patrol Availability",
                            value:
                              available !== null
                                ? `${available}${total !== null ? ` / ${total}` : ""} units`
                                : null,
                            mono: true,
                            wide: true,
                          },
                          ...extraRows,
                        ]}
                      />
                    </Popup>
                  </Marker>
                );
              })}
            </LayerGroup>
          </LayersControl.Overlay>

          {/* ---------------- ACTIVE INCIDENTS ---------------- */}
          <LayersControl.Overlay checked name="Active Incidents">
            <LayerGroup>
              {incidents.map((incident) => {
                const lat = Number(incident.latitude);
                const lng = Number(incident.longitude);
                if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;

                const severity = norm(incident.severity);
                const urgent = severity === "CRITICAL" || severity === "HIGH";
                const highlighted =
                  selectedId === incident.incident_id ||
                  highlightedIncidentId === incident.incident_id;
                const departments = parseDepartments(incident.assigned_departments);

                return (
                  <Marker
                    key={`incident-${incident.incident_id}`}
                    position={[lat, lng]}
                    icon={getIcon("incident", { pulse: urgent, selected: highlighted })}
                    zIndexOffset={highlighted ? 1500 : 1000}
                    eventHandlers={{
                      click: () => handleIncidentClick(incident, lat, lng),
                    }}
                  >
                    <Popup className="enterprise-popup">
                      <PopupCard
                        accent="text-red-400"
                        kicker="Active Incident"
                        title={titleCase(incident.incident_type) || "Active Incident"}
                        rows={[
                          { label: "Incident ID", value: incident.incident_id, mono: true },
                          {
                            label: "Severity",
                            value: severity || "UNKNOWN",
                            mono: true,
                            className: SEVERITY_TEXT[severity] || "text-slate-300",
                          },
                          { label: "Status", value: titleCase(incident.status || "Active") },
                          { label: "Reported", value: formatDateTime(incident.reported_at), mono: true },
                          { label: "Location", value: incident.location, wide: true },
                          {
                            label: "Assigned Departments",
                            value: departments.length
                              ? departments.map((d) => d.name).join(" + ")
                              : "Police",
                            wide: true,
                          },
                        ]}
                      >
                        <button
                          type="button"
                          onClick={() => openIncident(incident.incident_id)}
                          className="mt-3 w-full text-[11px] font-bold uppercase tracking-wider py-1.5 rounded bg-red-500/10 text-red-300 border border-red-500/40 hover:bg-red-500/20 transition-colors"
                        >
                          View Incident →
                        </button>
                      </PopupCard>
                    </Popup>
                  </Marker>
                );
              })}
            </LayerGroup>
          </LayersControl.Overlay>

          {/* ---------------- CRIME INTELLIGENCE ---------------- */}
          <LayersControl.Overlay checked name="Crime Intelligence">
            <LayerGroup>
              {crimes.map((crime) => {
                const lat = Number(crime.latitude);
                const lng = Number(crime.longitude);
                if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;

                const severity = norm(crime.severity);

                return (
                  <Marker
                    key={`crime-${crime.id}`}
                    position={[lat, lng]}
                    icon={getIcon("crime")}
                    zIndexOffset={750}
                  >
                    <Popup className="enterprise-popup">
                      <PopupCard
                        accent="text-purple-400"
                        kicker="Crime Record"
                        title={titleCase(crime.category) || "Crime Record"}
                        rows={[
                          { label: "Crime ID", value: crime.id, mono: true },
                          {
                            label: "Severity",
                            value: severity || "UNKNOWN",
                            mono: true,
                            className: SEVERITY_TEXT[severity] || "text-slate-300",
                          },
                          { label: "Status", value: titleCase(crime.status || "Reported") },
                          { label: "Reported", value: formatDateTime(crime.reported_at), mono: true },
                          { label: "Location", value: crime.location, wide: true },
                        ]}
                      />
                    </Popup>
                  </Marker>
                );
              })}
            </LayerGroup>
          </LayersControl.Overlay>
        </LayersControl>
      </MapContainer>

      {/* LEGEND (no counts) */}
      <div className="absolute bottom-3 left-3 z-[1000] bg-slate-950/90 border border-slate-700 rounded-md px-2.5 py-1.5 shadow-lg flex flex-col sm:flex-row gap-x-3 gap-y-1 text-[10px] text-slate-300">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-blue-500" /> Police Station
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-red-500" /> Active Incident
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-purple-500" /> Crime
        </span>
      </div>

      {/* Error only; no numeric status bar */}
      {error && (
        <div className="absolute top-3 left-3 z-[1000] bg-red-950/90 border border-red-500/40 text-red-300 text-[11px] px-3 py-1.5 rounded-md">
          Map data unavailable. Retrying…
        </div>
      )}
    </div>
  );
}