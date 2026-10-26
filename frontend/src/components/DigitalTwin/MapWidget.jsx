import React, { useEffect, useState } from "react";
import L from 'leaflet'; 
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import MarkerClusterGroup from "react-leaflet-cluster";
import { useMapStore } from "../../store/useMapStore";
import { getEnterpriseIcon } from "./iconHelper";

// 1. Interactive Legend & Layer Config
const LAYER_INDICATORS = [
  { id: 'incidents', label: 'Critical Incidents', color: 'bg-red-600' },
  { id: 'health', label: 'Hospitals / EMS', color: 'bg-red-500' },
  { id: 'police', label: 'Police Stations', color: 'bg-blue-600' },
  { id: 'traffic', label: 'Traffic & CCTVs', color: 'bg-amber-500' },
  { id: 'utilities', label: 'Water & Power', color: 'bg-cyan-500' },
  { id: 'fire', label: 'Fire Departments', color: 'bg-orange-500' },
  { id: 'weather', label: 'Weather Nodes', color: 'bg-sky-500' },
];

// Extracted Popup for clean architecture
const EnterprisePopup = ({ title, type, metadata }) => (
  <div className="w-64 bg-slate-900 text-slate-100 p-3 rounded-lg shadow-xl font-sans">
    <h3 className="font-bold text-md border-b border-slate-700 pb-2 mb-2 truncate text-blue-400">
      {title}
    </h3>
    <p className="text-xs text-slate-400 uppercase tracking-wider mb-3">
      Type: {type.replace("_", " ")}
    </p>
    <div className="grid grid-cols-2 gap-2 text-sm">
      {Object.entries(metadata).map(([key, val]) => (
        <div key={key} className="flex flex-col bg-slate-800 p-1.5 rounded">
          <span className="text-slate-400 text-xs capitalize">
            {key.replace(/([A-Z])/g, " $1")}
          </span>
          <span className="font-mono text-slate-100">{val}</span>
        </div>
      ))}
    </div>
  </div>
);

const createCustomClusterIcon = (cluster) => {
    // Get the number of markers grouped in this cluster
    const count = cluster.getChildCount();
    
    return L.divIcon({
        html: `<div class="enterprise-cluster">${count}</div>`,
        className: 'custom-cluster-icon',
        iconSize: [45, 45],
        iconAnchor: [22, 22] 
    });
};

// Critical incident marker — Now accepts `isSelected` to differentiate the active target
// Critical incident marker — Keeps red color, scales up when selected
const createIncidentIcon = (severity, isSelected) => {
  const isCritical = severity === 'Critical' || severity === 'High';
  
  // Scale up if selected, but keep standard sizes otherwise
  const ringSize = isSelected ? 68 : (isCritical ? 48 : 36);
  const dotSize = isSelected ? 40 : (isCritical ? 30 : 22);
  const glyphSize = isSelected ? 20 : (isCritical ? 16 : 12);

  // Keep colors strictly RED
  const glowColor = 'rgba(220, 38, 38, 0.45)';
  const bgColor = '#dc2626';

  return L.divIcon({
    className: 'custom-incident-marker',
    html: `
      <div style="position: relative; width: ${ringSize}px; height: ${ringSize}px; display: flex; align-items: center; justify-content: center;">
        <div style="position: absolute; width: ${ringSize}px; height: ${ringSize}px; background: ${glowColor}; border-radius: 50%; animation: ping 1.4s cubic-bezier(0, 0, 0.2, 1) infinite;"></div>
        <div style="width: ${dotSize}px; height: ${dotSize}px; background: ${bgColor}; border: 2px solid #ffffff; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 16px ${glowColor}; z-index: ${isSelected ? 100 : 10};">
          <svg style="width: ${glyphSize}px; height: ${glyphSize}px; fill: white;" viewBox="0 0 24 24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>
        </div>
      </div>
    `,
    iconSize: [ringSize, ringSize],
    iconAnchor: [ringSize / 2, ringSize / 2],
    popupAnchor: [0, -ringSize / 2],
  });
};

// FIX: Track primitive coordinates AND selectedIncidentId to force map to fly on every click
const MapViewController = ({ center, zoom, selectedIncidentId }) => {
  const map = useMap();
  const lat = center ? center[0] : null;
  const lng = center ? center[1] : null;

  useEffect(() => {
    if (lat !== null && lng !== null) {
      map.flyTo([lat, lng], zoom, { duration: 1.5 });
    }
  }, [lat, lng, zoom, selectedIncidentId, map]);

  return null;
};

// Page-specific glowing demo camera marker icon
const demoCameraIcon = L.divIcon({
  className: 'custom-demo-marker',
  html: `
    <div style="position: relative; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center;">
      <div style="position: absolute; width: 48px; height: 48px; background: rgba(59, 130, 246, 0.45); border-radius: 50%; animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;"></div>
      <div style="width: 32px; height: 32px; background: #3b82f6; border: 2px solid #ffffff; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(59, 130, 246, 0.9); z-index: 10;">
        <svg style="width: 16px; height: 16px; fill: white;" viewBox="0 0 24 24"><path d="M15 8v8H5V8h10m1-2H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4V7c0-.55-.45-1-1-1z"/></svg>
      </div>
    </div>
  `,
  iconSize: [36, 36],
  iconAnchor: [18, 18],
  popupAnchor: [0, -20]
});

// Phase 2/3: Added focusedCenter, focusedZoom, and selectedIncidentId props
const DigitalTwinMap = ({ demoLocation, focusedCenter, focusedZoom, selectedIncidentId }) => {
  const { 
    getFilteredEntities, 
    vehicles, 
    incidents, 
    initializeMap,
    activeLayers,
    toggleLayer
  } = useMapStore();
  
  const visibleEntities = getFilteredEntities();
  const [showControls, setShowControls] = useState(false);

  useEffect(() => {
    fetch("/api/map/data")
      .then((res) => res.json())
      .then((data) => initializeMap(data));
  }, [initializeMap]);

  const defaultCenter = [19.076, 72.8777];
  
  // Prioritize explicitly focused center/zoom from Dashboard, fallback to demoLocation, then default
  const currentCenter = focusedCenter || (demoLocation ? [demoLocation.lat, demoLocation.lng] : defaultCenter);
  const currentZoom = focusedZoom || (demoLocation ? 17 : 12);

  return (
    <div className="relative w-full h-full bg-slate-950 overflow-hidden">
      
      {/* Floating Layer Buttons & Icon Indicator Panel */}
      <div className="absolute top-4 right-4 z-[1000] flex flex-col items-end gap-2">
        <button 
          onClick={() => setShowControls(!showControls)}
          className="bg-slate-900/90 text-white px-4 py-2 rounded-lg border border-slate-700 shadow-lg hover:bg-slate-800 transition-colors backdrop-blur-md font-bold text-sm"
        >
          {showControls ? 'Hide Layers' : 'Show Layers & Legend'}
        </button>

        {showControls && (
          <div className="w-64 bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-xl shadow-2xl p-4 text-slate-200">
            <h4 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-3 border-b border-slate-700 pb-2">
              Map Layers & Legend
            </h4>
            <div className="flex flex-col gap-2">
              {LAYER_INDICATORS.map((layer) => {
                const isActive = activeLayers[layer.id] !== false; // Default to true if undefined
                return (
                  <button
                    key={layer.id}
                    onClick={() => toggleLayer(layer.id)}
                    className={`flex items-center justify-between w-full p-2 rounded-lg transition-all duration-200 ${
                      isActive ? 'bg-slate-800 hover:bg-slate-700' : 'bg-slate-950/50 opacity-60 grayscale'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      {/* Icon Color Indicator */}
                      <span className={`w-3 h-3 rounded-full ${layer.color} shadow-[0_0_8px_currentColor] ${layer.id === 'incidents' ? 'animate-pulse' : ''}`}></span>
                      <span className="text-sm font-medium">{layer.label}</span>
                    </div>
                    
                    {/* Layer Toggle Switch */}
                    <div className={`w-8 h-4 rounded-full relative transition-colors ${isActive ? 'bg-blue-600' : 'bg-slate-600'}`}>
                      <div className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform ${isActive ? 'translate-x-4' : 'translate-x-0'}`}></div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <MapContainer
        center={defaultCenter}
        zoom={12}
        className="absolute inset-0 z-0"
        zoomControl={false}
      >
        <MapViewController center={currentCenter} zoom={currentZoom} selectedIncidentId={selectedIncidentId} />

        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Optional Page-Specific Demonstration Marker */}
        {demoLocation && (
          <Marker position={[demoLocation.lat, demoLocation.lng]} icon={demoCameraIcon}>
            <Popup className="enterprise-popup border-t-4 border-blue-500">
              <div className="w-64 bg-slate-900 text-slate-100 p-3 rounded-lg shadow-xl font-sans">
                <h3 className="font-bold text-md border-b border-slate-700 pb-1 text-blue-400 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span> Demonstration Camera
                </h3>
                <div className="mt-2 space-y-1 text-xs font-mono">
                  <div><span className="text-slate-400">Camera ID:</span> <span className="text-white font-bold">{demoLocation.id}</span></div>
                  <div><span className="text-slate-400">Location:</span> <span className="text-white">{demoLocation.name}</span></div>
                  <div><span className="text-slate-400">Latitude:</span> <span className="text-slate-300">{demoLocation.lat}</span></div>
                  <div><span className="text-slate-400">Longitude:</span> <span className="text-slate-300">{demoLocation.lng}</span></div>
                  <div className="pt-2 border-t border-slate-800 flex justify-between items-center">
                    <span className="text-slate-400">Status:</span>
                    <span className="bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/30 text-[10px] font-bold">Ready</span>
                  </div>
                </div>
              </div>
            </Popup>
          </Marker>
        )}

        {/* Layer 1: Static Clustered Entities (Hospitals, Police, Sensors) */}
        <MarkerClusterGroup chunkedLoading maxClusterRadius={30} iconCreateFunction={createCustomClusterIcon} disableClusteringAtZoom={8}>
          {visibleEntities.map((entity) => (
            <Marker
              key={entity.entity_id}
              position={[
                parseFloat(entity.latitude),
                parseFloat(entity.longitude),
              ]}
              icon={getEnterpriseIcon(entity.entity_type)}
            >
              <Popup className="enterprise-popup">
                <EnterprisePopup
                  title={entity.name}
                  type={entity.entity_type}
                  metadata={entity.metadata}
                />
              </Popup>
            </Marker>
          ))}
        </MarkerClusterGroup>

        {/* Layer 2: Real-time Moving Vehicles */}
        {vehicles.map((v) => (
          <Marker
            key={v.vehicle_id}
            position={[parseFloat(v.current_lat), parseFloat(v.current_lng)]}
            icon={getEnterpriseIcon(v.vehicle_type)}
            eventHandlers={{
              add: (e) =>
                e.target
                  .getElement()
                  .classList.add("transition-all", "duration-1000"),
            }}
          >
            <Popup className="enterprise-popup">
              <EnterprisePopup
                title={`${v.department} Unit`}
                type={v.vehicle_type}
                metadata={{ status: v.status }}
              />
            </Popup>
          </Marker>
        ))}

        {/* Layer 3: Active Incidents (Highest Z-Index, Pulses) */}
        {/* FIX: Renders ALL active incidents, bringing the selected target to the front with a blue glow */}
        {activeLayers['incidents'] !== false && incidents
          .filter(inc => inc.status && inc.status.toLowerCase() !== 'resolved' && inc.status.toLowerCase() !== 'inactive')
          .map((inc) => {
            const isSelected = String(inc.incident_id) === String(selectedIncidentId);
            return (
              <Marker
                key={inc.incident_id}
                position={[parseFloat(inc.latitude), parseFloat(inc.longitude)]}
                icon={createIncidentIcon(inc.severity, isSelected)}
                zIndexOffset={isSelected ? 1000 : 0}
              >
                <Popup className={`enterprise-popup border-t-4 ${isSelected ? 'border-blue-500' : 'border-red-500'}`}>
                  <EnterprisePopup
                    title={inc.incident_type}
                    type="Critical Alert"
                    metadata={{ severity: inc.severity, status: inc.status }}
                  />
                </Popup>
              </Marker>
            );
          })}
      </MapContainer>
    </div>
  );
};

export default DigitalTwinMap;