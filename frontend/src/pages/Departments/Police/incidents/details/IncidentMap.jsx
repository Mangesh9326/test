import React, { useEffect, useState } from "react";
import { MapContainer, Marker, Popup, TileLayer, ZoomControl } from "react-leaflet";
import { getIcon } from "./mapIcons";
import { formatDateTime, titleCase } from "../utils";

const ZOOM = 14;

export default function IncidentMap({ incident, stations, crimes, recenterKey }) {
  const [map, setMap] = useState(null);
  const lat = Number(incident.latitude);
  const lng = Number(incident.longitude);
  const hasCoords = Number.isFinite(lat) && Number.isFinite(lng);

  useEffect(() => {
    if (map && hasCoords && recenterKey) map.flyTo([lat, lng], 15, { duration: 0.8 });
  }, [recenterKey]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!hasCoords) {
    return (
      <div className="h-[260px] bg-slate-900 border border-slate-700 rounded-lg flex items-center justify-center text-xs text-slate-500">
        Location coordinates are not available for this incident.
      </div>
    );
  }

  return (
    <div id="incident-map" className="relative h-[360px] sm:h-[460px] bg-slate-950 border border-slate-700 rounded-lg overflow-hidden shadow-lg">
      <style>{`@keyframes incident-map-ping{75%,100%{transform:scale(1.9);opacity:0}}`}</style>

      <MapContainer key={incident.id} center={[lat, lng]} zoom={ZOOM} zoomControl={false} ref={setMap} className="absolute inset-0 z-0">
        <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" className="dark-map-tiles" />
        <ZoomControl position="bottomright" />

        {stations.map((s) => {
          const sLat = Number(s.latitude), sLng = Number(s.longitude);
          if (!Number.isFinite(sLat) || !Number.isFinite(sLng)) return null;
          return (
            <Marker key={`st-${s.entity_id}`} position={[sLat, sLng]} icon={getIcon("station")} zIndexOffset={500}>
              <Popup className="enterprise-popup">
                <div className="w-52 bg-slate-900 text-slate-100 p-2.5 rounded">
                  <div className="text-[9px] uppercase tracking-widest text-blue-400">Police Station</div>
                  <div className="text-sm font-bold">{s.name}</div>
                  {s.distance_km != null && <div className="text-[10px] font-mono text-slate-400">{Number(s.distance_km).toFixed(1)} km away</div>}
                </div>
              </Popup>
            </Marker>
          );
        })}

        {crimes.map((c) => {
          const cLat = Number(c.latitude), cLng = Number(c.longitude);
          if (!Number.isFinite(cLat) || !Number.isFinite(cLng)) return null;
          return (
            <Marker key={`cr-${c.id}`} position={[cLat, cLng]} icon={getIcon("crime")} zIndexOffset={750}>
              <Popup className="enterprise-popup">
                <div className="w-52 bg-slate-900 text-slate-100 p-2.5 rounded">
                  <div className="text-[9px] uppercase tracking-widest text-purple-400">Crime Record</div>
                  <div className="text-sm font-bold">{titleCase(c.category)}</div>
                  <div className="text-[10px] text-slate-400">{c.location}</div>
                  <div className="text-[10px] font-mono text-slate-500">{formatDateTime(c.reported_at)}</div>
                </div>
              </Popup>
            </Marker>
          );
        })}

        <Marker position={[lat, lng]} icon={getIcon("incident", incident.severityKey === "CRITICAL" || incident.severityKey === "HIGH")} zIndexOffset={2000}>
          <Popup className="enterprise-popup">
            <div className="w-52 bg-slate-900 text-slate-100 p-2.5 rounded">
              <div className="text-[9px] uppercase tracking-widest text-red-400">Incident</div>
              <div className="text-sm font-bold">{incident.typeLabel}</div>
              <div className="text-[10px] text-slate-400">{incident.location}</div>
            </div>
          </Popup>
        </Marker>
      </MapContainer>

      <button
        type="button"
        onClick={() => map?.flyTo([lat, lng], 15, { duration: 0.8 })}
        className="absolute top-3 right-3 z-[1000] text-[10px] font-bold uppercase tracking-wider px-2.5 py-1.5 rounded border border-slate-600 bg-slate-950/90 text-slate-200 hover:bg-slate-800"
      >
        Recenter
      </button>

      <div className="absolute bottom-3 left-3 z-[1000] bg-slate-950/90 border border-slate-700 rounded-md px-2.5 py-1.5 flex gap-3 text-[10px] text-slate-300">
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-red-500" />Incident</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-blue-500" />Station</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-purple-500" />Crime</span>
      </div>
    </div>
  );
}
